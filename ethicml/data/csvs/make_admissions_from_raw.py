"""Add column names to the CSV from Harvard dataverse 'UFRGS Entrance Exam and GPA Data'.

Persistent link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O35FW8

Data Description: Entrance exam scores of students applying to a university in Brazil
(Federal University of Rio Grande do Sul), along with the students' GPAs during the first three
semesters at university. In this dataset, each row contains anonymized information about an
applicant's scores on nine exams taken as part of the application process to the university, as
well as their corresponding GPA during the first three semesters at university.
The dataset has 43,303 rows, each corresponding to one student.
The columns correspond to:
1) Gender. 0 denotes female and 1 denotes male.
2) Score on physics exam
3) Score on biology exam
4) Score on history exam
5) Score on second language exam
6) Score on geography exam
7) Score on literature exam
8) Score on Portuguese essay exam
9) Score on math exam
10) Score on chemistry exam
11) Mean GPA during first three semesters at university, on a 4.0 scale.

We replace the mean GPA with a binary label Y representing whether the student’s GPA was above 3.0


```bibtex
@data{DVN/O35FW8_2019,
  author    = {Castro~da~Silva, Bruno},
  publisher = {Harvard Dataverse},
  title     = {{UFRGS Entrance Exam and GPA Data}},
  UNF       = {UNF:6:MQqEQGXiIfQTbS7q9QJ5uw==},
  year      = {2019},
  version   = {V1},
  doi       = {10.7910/DVN/O35FW8},
  url       = {https://doi.org/10.7910/DVN/O35FW8}
}
```
"""


import pandas as pd


def run_generate_admissions() -> None:
    """Add a the header solumns, binarise label and shuffle."""
    data = pd.read_csv("raw/ufrgs_entrance_exam_and_gpa.csv.zip", header=None)

    # Give data column names
    columns = [
        "gender",
        "physics",
        "biology",
        "history",
        "language",
        "geography",
        "literature",
        "essay",
        "math",
        "chemistry",
        "gpa",
    ]
    data.columns = pd.Index(columns)

    # Binarize the GPA column
    data["gpa"] = data["gpa"] >= 3.0
    data["gpa"] = data["gpa"].astype("int")

    # Shuffle the data
    data = data.sample(frac=1.0, random_state=888).reset_index(drop=True)

    # Save the CSV
    compression_opts = dict(method='zip', archive_name='admissions.csv')
    data.to_csv("./admissions.csv.zip", index=False, compression=compression_opts)


if __name__ == '__main__':
    run_generate_admissions()
