# Chess Dataset

The analyses expect a BIDS formatted dataset with the following structure:

```
/data/projects/chess/data/
├── BIDS/
│   ├── sub-01/
│   ├── sub-02/
│   └── ...
└── derivatives/
    ├── fmriprep/
    ├── mvpa/
    └── fastsurfer/
```

Place raw functional and anatomical data under `BIDS/` and ensure that the derivatives produced by fMRIprep and FastSurfer are available under `derivatives/`. The MATLAB and Python scripts reference this path by default. Modify `modules.BASE_DATA_PATH` in `chess-mvpa/modules/__init__.py` if your dataset is located elsewhere.

Example JSON metadata files are included in the `misc/` directory to demonstrate the expected key/value pairs for the BIDS sidecars.
