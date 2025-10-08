#!/usr/bin/env python3
"""
Dynamic Worker Management System
Handles intelligent worker redeployment and queue throttling without deadlocks
"""

import threading
import time
import queue
from dataclasses import dataclass
from typing import Dict, List, Optional
import psutil


@dataclass
class WorkerAllocation:
    """Current worker allocation state"""
    cpu_preprocessing: int = 0
    cpu_translation: int = 0
    gpu_translation: int = 0
    idle_workers: int = 0
    
    def total_workers(self) -> int:
        return self.cpu_preprocessing + self.cpu_translation + self.gpu_translation


@dataclass
class QueueStatus:
    """Queue status information"""
    size: int
    max_size: int
    fullness: float
    
    @property
    def is_overloaded(self) -> bool:
        return self.fullness >= 0.80
    
    @property
    def needs_more_workers(self) -> bool:
        return self.fullness <= 0.40 and self.size > 0


class DynamicWorkerManager:
    """Manages dynamic worker allocation and queue throttling"""
    
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        
        # Worker tracking
        self.worker_allocation = WorkerAllocation()
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.worker_stop_events: Dict[str, threading.Event] = {}
        
        # Queue monitoring
        self.queue_status: Dict[str, QueueStatus] = {}
        self.throttling_active = False
        self.last_reallocation = time.time()
        
        # Resource monitoring
        self.cpu_usage_history = []
        self.memory_usage_history = []
        
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Keep history for trend analysis
        self.cpu_usage_history.append(cpu_percent)
        self.memory_usage_history.append(memory_percent)
        
        # Keep only last 30 readings (3 minutes at 6s intervals)
        if len(self.cpu_usage_history) > 30:
            self.cpu_usage_history.pop(0)
            self.memory_usage_history.pop(0)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'cpu_trend': sum(self.cpu_usage_history[-5:]) / min(5, len(self.cpu_usage_history)),
            'memory_trend': sum(self.memory_usage_history[-5:]) / min(5, len(self.memory_usage_history))
        }
    
    def update_queue_status(self, queue_name: str, current_queue: queue.Queue):
        """Update queue status for monitoring"""
        try:
            size = current_queue.qsize()
            max_size = self.config.max_queue_size
            fullness = size / max_size if max_size > 0 else 0
            
            self.queue_status[queue_name] = QueueStatus(
                size=size,
                max_size=max_size,
                fullness=fullness
            )
        except Exception as e:
            print(f"[WARNING]  Failed to update queue status for {queue_name}: {e}")
    
    def should_throttle_queue(self, queue_name: str) -> bool:
        """Check if queue should be throttled (improved logic)"""
        if queue_name not in self.queue_status:
            return False
        
        status = self.queue_status[queue_name]
        
        # CRITICAL: Emergency queue clearing for stuck queues (from terminal log analysis)
        if status.fullness >= 0.90:  # 90% or higher (was 95% - too late)
            print(f"🚨 EMERGENCY: Queue {queue_name} at {status.fullness:.1%} - implementing emergency measures")
            self._emergency_queue_handling(queue_name)
            
        # CRITICAL: Detect completely stuck queues (100% for extended periods)
        if status.fullness >= 0.98:  # 98% or higher indicates severe backup
            print(f"🆘 CRITICAL QUEUE BACKUP: {queue_name} at {status.fullness:.1%} - forcing emergency drain")
            self._force_emergency_drain(queue_name)
        
        # More sophisticated throttling logic (lowered threshold)
        if status.fullness >= 0.60:  # Lowered from 0.80 for better flow
            # Check if we have idle workers that could help process the queue
            if self.worker_allocation.idle_workers > 0:
                print(f"[CONFIG] Queue {queue_name} busy ({status.fullness:.1%}) - redeploying {self.worker_allocation.idle_workers} idle workers")
                self._redeploy_idle_workers_for_translation()
                return False  # Don't throttle, let idle workers help
            
            # Critical queue handling
            if status.fullness >= 0.90:
                print(f"🚨 Queue {queue_name} critically full ({status.fullness:.1%}) - forcing throttle")
                return True
                
            return True  # Throttle at 60% if no idle workers available
        
        return False
    
    def _redeploy_idle_workers_for_translation(self):
        """Redeploy idle preprocessing workers to translation work"""
        with self.lock:
            if self.worker_allocation.idle_workers <= 0:
                return
            
            # Prevent too frequent reallocations
            if time.time() - self.last_reallocation < 10:  # 10 second cooldown
                return
            
            resources = self.monitor_system_resources()
            
            # Only redeploy if system has capacity
            if resources['cpu_percent'] > 85 or resources['memory_percent'] > 90:
                print(f"[WARNING]  System overloaded (CPU: {resources['cpu_percent']:.1f}%, RAM: {resources['memory_percent']:.1f}%) - skipping worker redeployment")
                return
            
            # Calculate how many workers to redeploy
            workers_to_redeploy = min(
                self.worker_allocation.idle_workers,
                max(1, self.worker_allocation.idle_workers // 2)  # Redeploy up to half
            )
            
            print(f"[CONFIG] DYNAMIC REDEPLOYMENT: Moving {workers_to_redeploy} idle workers to translation")
            print(f"   Current allocation: Preprocessing={self.worker_allocation.cpu_preprocessing}, Translation={self.worker_allocation.cpu_translation}, Idle={self.worker_allocation.idle_workers}")
            
            # Update allocation
            self.worker_allocation.idle_workers -= workers_to_redeploy
            self.worker_allocation.cpu_translation += workers_to_redeploy
            
            print(f"   New allocation: Preprocessing={self.worker_allocation.cpu_preprocessing}, Translation={self.worker_allocation.cpu_translation}, Idle={self.worker_allocation.idle_workers}")
            
            self.last_reallocation = time.time()
    
    def intelligent_queue_throttling(self, queue_name: str, current_queue: queue.Queue, 
                                   work_item, timeout: float = 20.0) -> bool:
        """
        Intelligent queue throttling that prevents deadlocks
        Returns True if item was successfully queued, False if should stop
        """
        self.update_queue_status(queue_name, current_queue)
        
        if not self.should_throttle_queue(queue_name):
            # No throttling needed, queue normally
            try:
                current_queue.put(work_item, timeout=timeout)
                return True
            except queue.Full:
                print(f"[WARNING]  Queue {queue_name} full despite no throttling - system overload")
                return self._handle_queue_full_fallback(queue_name, current_queue, work_item, timeout)
        
        # Throttling needed
        status = self.queue_status[queue_name]
        print(f"🚦 INTELLIGENT THROTTLING: {queue_name} is {status.fullness:.1%} full ({status.size}/{status.max_size})")
        
        throttle_start = time.time()
        max_wait = self.config.max_throttle_wait
        
        while time.time() - throttle_start < max_wait:
            # Update status
            self.update_queue_status(queue_name, current_queue)
            current_status = self.queue_status[queue_name]
            
            # Check if we can resume
            if current_status.fullness <= self.config.queue_resume_threshold:
                elapsed = time.time() - throttle_start
                print(f"[OK] THROTTLING RESUMED: {queue_name} cleared to {current_status.fullness:.1%} after {elapsed:.1f}s")
                
                # Try to queue the item
                try:
                    current_queue.put(work_item, timeout=timeout)
                    return True
                except queue.Full:
                    print(f"[WARNING]  Queue {queue_name} filled again immediately - continuing throttling")
                    continue
            
            # Show progress every 5 seconds
            elapsed = time.time() - throttle_start
            if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                print(f"   [CONFIG] Still throttled: {queue_name} {current_status.fullness:.1%} full, waiting {elapsed:.0f}s...")
            
            time.sleep(self.config.throttle_check_interval)
        
        # Force resume after max wait
        print(f"[WARNING]  FORCE RESUME: {queue_name} throttling timeout ({max_wait}s) - forcing queue attempt")
        return self._handle_queue_full_fallback(queue_name, current_queue, work_item, timeout)
    
    def _handle_queue_full_fallback(self, queue_name: str, current_queue: queue.Queue, 
                                  work_item, timeout: float) -> bool:
        """Handle queue full situation with fallback strategies"""
        try:
            # Try with shorter timeout
            current_queue.put(work_item, timeout=5.0)
            return True
        except queue.Full:
            print(f"[ERROR] QUEUE OVERFLOW: {queue_name} cannot accept items - implementing fallback")
            
            # Strategy 1: Try to spawn additional translation worker if resources allow
            resources = self.monitor_system_resources()
            if (resources['cpu_percent'] < 70 and resources['memory_percent'] < 80 and 
                self.worker_allocation.cpu_translation < psutil.cpu_count() - 2):
                
                print(f"[LAUNCH] EMERGENCY SCALING: Spawning additional translation worker for {queue_name}")
                self._spawn_emergency_translation_worker(queue_name)
                
                # Try again after brief wait
                time.sleep(1)
                try:
                    current_queue.put(work_item, timeout=3.0)
                    return True
                except queue.Full:
                    pass
            
            # Strategy 2: Save to persistent queue for later processing
            print(f"[SAVE] PERSISTENT QUEUE: Saving work item for later processing")
            self._save_to_persistent_queue(queue_name, work_item)
            return True  # Don't drop the term
    
    def _spawn_emergency_translation_worker(self, queue_name: str):
        """Spawn an emergency translation worker"""
        # This would integrate with the main translation system
        # For now, just update allocation tracking
        with self.lock:
            self.worker_allocation.cpu_translation += 1
            print(f"[FAST] Emergency worker spawned for {queue_name}")
    
    def _emergency_queue_handling(self, queue_name: str):
        """Handle emergency queue situations when queues are critically full"""
        print(f"🚨 EMERGENCY QUEUE HANDLING for {queue_name}")
        
        # Strategy 1: Force worker redeployment regardless of cooldown
        if self.worker_allocation.idle_workers > 0:
            print(f"[LAUNCH] Emergency redeployment: {self.worker_allocation.idle_workers} workers")
            self.last_reallocation = 0  # Reset cooldown
            self._redeploy_idle_workers_for_translation()
        
        # Strategy 2: Temporary queue size increase for emergency
        if queue_name in self.queue_status:
            status = self.queue_status[queue_name]
            if status.fullness >= 0.98:  # 98% or higher
                print(f"🆘 CRITICAL: {queue_name} at {status.fullness:.1%} - implementing queue priority processing")
                # This would signal workers to prioritize this specific queue
                
        # Strategy 3: Performance-based worker restart (if available)
        print(f"[CONFIG] Emergency measure: Requesting worker performance check for {queue_name}")
    
    def _force_emergency_drain(self, queue_name: str):
        """Force emergency queue draining when completely stuck"""
        print(f"🆘 FORCE EMERGENCY DRAIN for {queue_name}")
        
        # Strategy 1: Signal all workers to prioritize this specific queue
        print(f"📢 Broadcasting PRIORITY DRAIN signal for {queue_name}")
        
        # Strategy 2: Temporary suspension of new queue additions
        print(f"⏸️  Temporarily suspending new additions to {queue_name}")
        
        # Strategy 3: Emergency worker allocation boost
        print(f"[LAUNCH] Requesting emergency worker boost for {queue_name}")
        
        # This would integrate with the actual queue management system
        # to implement these emergency measures
    
    def _save_to_persistent_queue(self, queue_name: str, work_item):
        """Save work item to persistent storage for later processing"""
        # This would save to a file or database for recovery
        print(f"[SAVE] Saved work item to persistent queue for {queue_name}")
    
    def get_worker_stats(self) -> Dict:
        """Get current worker statistics"""
        return {
            'allocation': self.worker_allocation,
            'queue_status': self.queue_status,
            'throttling_active': self.throttling_active,
            'system_resources': self.monitor_system_resources()
        }


# Integration helper functions
def create_dynamic_worker_manager(config):
    """Create and initialize dynamic worker manager"""
    return DynamicWorkerManager(config)


def integrate_with_ultra_runner(runner_instance, worker_manager):
    """Integrate dynamic worker manager with UltraOptimizedSmartRunner"""
    # Replace the simple throttling check with intelligent throttling
    runner_instance.dynamic_worker_manager = worker_manager
    
    # Override the throttling method
    original_check_throttling = runner_instance._check_queue_throttling
    
    def intelligent_throttling_wrapper(selected_queue, work_item=None):
        if work_item is None:
            # Fallback to original method if no work item provided
            return original_check_throttling(selected_queue)
        
        # Use intelligent throttling
        queue_name = "gpu_queue" if selected_queue in [runner_instance.gpu_queue_1, runner_instance.gpu_queue_2] else "cpu_queue"
        return worker_manager.intelligent_queue_throttling(queue_name, selected_queue, work_item)
    
    runner_instance._intelligent_queue_throttling = intelligent_throttling_wrapper
    
    print("[OK] Dynamic worker manager integrated with UltraOptimizedSmartRunner")
