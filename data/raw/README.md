# JDsearch dataset

The two raw dataset files are not stored in this GitHub repository because of their size.

| File | Approximate size |
|---|---:|
| `user_behavior_data.txt` | 0.96 GB |
| `product_meta_data.txt` | 4.84 GB |

## Download the data

Download both files from either:

- [Team’s shared Google Drive folder](https://drive.google.com/drive/folders/1BndBsCZtrcNvvq-NkYt--LMIR-2PHHfu?usp=sharing)
- [Official JDsearch repository](https://github.com/rucliujn/JDsearch), which provides the JD Cloud download link and password in its README.

Ensure that anyone reviewing the project has permission to access the Google Drive folder.

## Set up the files

Place both files in the `data/raw/` directory:

```text
data/
└── raw/
    ├── user_behavior_data.txt
    └── product_meta_data.txt
```

If the files are stored elsewhere, update `paths.raw_dir` in `configs/default.yaml` or provide the path as a configuration override.

For Google Colab, place the files in `MyDrive/bt4222/raw/` as described in the main repository README.
