# Project Name

This project is implemented in Python 3.8 and involves tasks such as atomic point detection, feature extraction, dimensionality reduction, clustering, and visualization. The analysis and processing are done using Jupyter Notebooks.

## Installing Dependencies

Before using this project, ensure that Python 3.8 is installed and the required dependencies are in place:

1. Clone the repository:

   ```bash
   git clone <git repo>
   cd <repo path>
   ```

2. Create and activate a virtual environment (optional, but recommended):

   ```bash
   conda create -n atom python=3.8
   conda activate atom
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Task Workflow

This project performed using Jupyter Notebooks.

### Stage 1: Point Detection

Run `Stage1-Point-Detection.ipynb` for atomic point detection.

1. Open `Stage1-Point-Detection.ipynb` and run all the code cells.
2. This stage will output the coordinates or relevant information for the detected atomic points.

### Stage 2: Feature Extraction, Dimensionality Reduction, Clustering, and Visualization

After completing point detection, run `Stage2-ZP-Kmeans.ipynb` for atomic feature extraction, dimensionality reduction, clustering, and visualization.

1. Open `Stage2-ZP-Kmeans.ipynb` and run all the code cells.
2. In this stage, you will:
   - Extract atomic features;
   - Perform dimensionality reduction;
   - Use the K-means clustering algorithm for clustering;
   - Visualize the clustering results.

## File Structure

```
your_project_directory/
│
├── Stage1-Point-Detection.ipynb  # Atomic point detection task
├── Stage2-ZP-Kmeans.ipynb        # Atomic feature extraction, dimensionality reduction, clustering, and visualization
├── requirements.txt              # Dependencies file
├── task1                         # task1
├── task2                         # task2
└── README.md                     # Project documentation
```

## License

This project is licensed under the [MIT License](LICENSE).
