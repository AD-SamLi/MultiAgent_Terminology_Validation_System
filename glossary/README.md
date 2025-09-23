# Glossary

## Glossary csv File Guidelines

- The CSV file must be encoded in UTF-8.
- The first line must be `source,target`. **Note:** this is case-sensitive.
- Ensure there are no empty lines.
- Avoid leading or trailing spaces in both source and target columns.
- Use regular spaces, not no-break spaces, unless intentional.
- Find a sample CSV file [here](sample.csv).

## Folder Structure

- `data/general/ACAD`: ACAD glossary
- `data/general/AEC`: AEC glossary
- `data/general/CORE`: CORE glossary
- `data/general/DNM`: DNM glossary
- `data/general/MNE`: MNE glossary

## Managing Revisions

For example, there are 2 csv files `ACAD_CHS_terms_2024-08-19` and `AEC_CHS_terms_2024-08-19` created on `2024-08-19`.

- Create a new branch `glossary-2024-08-19`.
- Copy the csv files to the corresponding folder. **Note:** Do not add the time stamp to the file name.
  - `/data/glossary/ACAD/ACAD_CHS_terms.csv`
  - `/data/glossary/AEC/AEC_CHS_terms.csv`
- Create a new commit with message `Glossary on 2024-08-19`.
- Push to the repository.
- Create a pull request to merge the branch `glossary-2024-08-19` to the `main` branch.
- Do not merge the PR; inform Autodesk team to review the pull request.

## Update index.csv

Always update `index.csv` in each folder to clearly define the source and target language of the glossary file, here is an example of the `index.csv` file:

```csv
file,source,target
ACAD_CHS_terms.csv,en,zh-cn
ACAD_CHT_terms.csv,en,zh-tw
```
