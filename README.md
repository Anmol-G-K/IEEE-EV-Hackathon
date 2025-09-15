# 🚗 IEEE EV - Car Hacking Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.49.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/) [![Polars](https://img.shields.io/badge/Polars-1.33.1-0099FF?style=for-the-badge&logo=polars&logoColor=white)](https://pola.rs/) [![uv](https://img.shields.io/badge/uv-0.4.30-FF69B4?style=for-the-badge&logo=lightning&logoColor=white)](https://docs.astral.sh/uv/)



<!-- [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) -->


A comprehensive machine learning framework for detecting intrusions in automotive CAN bus networks using the **Car Hacking: Attack & Defense Challenge 2020** dataset from IEEE Dataport. This project implements cutting-edge deep learning approaches including Graph Neural Networks, Transformers, and hybrid architectures for automotive cybersecurity.

## 🎯 Overview

This project addresses the critical challenge of securing modern vehicles against cyber attacks by developing advanced intrusion detection systems (IDS) for Controller Area Network (CAN) bus communications. The framework implements multiple state-of-the-art machine learning approaches:

- **🔗 Graph Convolutional Networks (GCN)** for network topology-based anomaly detection
- **Hybrid ML Framework** combining sequence transformers, graph neural networks, and contrastive learning
- **🌳 Traditional ML Models** (Random Forest & XGBoost) for baseline comparison and ensemble methods
- **Interactive Streamlit Dashboard** for real-time analysis and model evaluation

## 📊 Dataset

The project utilizes the [Car Hacking: Attack & Defense Challenge 2020 Dataset](https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset) which contains:

### Dataset Statistics
- **Total Messages**: 8,694,507 CAN bus messages
- **Training Data**: 3,672,151 messages
- **Test Data**: 3,752,046 messages  
- **Validation Data**: 1,270,310 messages

### Attack Types
The dataset includes four main attack categories:

| Attack Type | Count | Description |
|-------------|-------|-------------|
| **Flooding** | 345,859 | High-frequency message injection attacks |
| **Fuzzing** | 216,571 | Random payload injection attacks |
| **Spoofing** | 200,338 | Message impersonation attacks |
| **Replay** | 110,474 | Previously captured message replay attacks |
| **Normal** | 7,821,265 | Legitimate vehicle communication |

### Data Structure
```bash
├───0_Preliminary/
│   ├───0_Training/ # Training Files
│   │       Pre_train_D_0.csv
│   │       Pre_train_D_1.csv
│   │       Pre_train_D_2.csv
│   │       Pre_train_S_0.csv
│   │       Pre_train_S_1.csv
│   │       Pre_train_S_2.csv
│   │
│   └───1_Submission/ # Test Files
│           Pre_submit_D.csv
│           Pre_submit_S.csv
│
└───1_Final/ # Validation Files
        Fin_host_session_submit_S.csv
```

### CAN Message Format
Each CSV contains CAN bus messages with the following structure:
- `Timestamp`: Unix timestamp of message transmission
- `Arbitration_ID`: CAN message identifier (hex format)
- `DLC`: Data Length Code (0-8 bytes)
- `Data`: Hexadecimal payload data (up to 16 hex characters)
- `Class`: Primary classification (Normal/Attack)
- `SubClass`: Detailed attack type (Normal/Flooding/Fuzzing/Spoofing/Replay)

## 🏗️ Project Architecture

```
.
│   .gitignore
│   .python-version
│   main.py # main file
│   pyproject.toml
│   README.md
│   uv.lock
│   
├───data
│   ├───0_Preliminary
│   │   ├───0_Training
│   │   │       Pre_train_D_0.csv
│   │   │       Pre_train_D_1.csv
│   │   │       Pre_train_D_2.csv
│   │   │       Pre_train_S_0.csv
│   │   │       Pre_train_S_1.csv
│   │   │       Pre_train_S_2.csv
│   │   │
│   │   └───1_Submission
│   │           Pre_submit_D.csv
│   │           Pre_submit_S.csv
│   │
│   └───1_Final
│           Fin_host_session_submit_S.csv
│
├───helpers
│       data_viewer.py
│       schema_viewer.py
│
├───out # EDA's
│   ├───eda_out
│   │       eda_summary.json
│   │       sample_head.csv
│   │
│   └───schema_debug
│           schema_report.json
│
├───src
│       ensemble_trial.py
│       GCNN.py # Graph Convolutional Neural Network (WIP) 
│       ML.py # ML implementations specifically Random Forest and XGBoost
│
└───utils
        can_ids_streamlit_app.py # An interactive dashboard for visualisation

```        


## 🚀 Quick Start

### Prerequisites

- **Python**: 3.13+ (specified in `.python-version`)
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **Memory**: 8GB+ RAM recommended for full dataset processing
- **Storage**: 2GB+ free space for dataset and outputs

### Installation

```bash
# Clone the repository
git clone https://github.com/Anmol-G-K/IEEE-EV-Hackathon.git
cd IEEE_EV

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | ≥2.8.0 | Deep learning framework |
| **torch-geometric** | Latest | Graph neural networks |
| **scikit-learn** | ≥1.7.2 | Traditional ML algorithms |
| **XGBoost** | ≥3.0.5 | Gradient boosting |
| **Polars** | ≥1.33.1 | Fast data processing |
| **Streamlit** | ≥1.49.1 | Interactive dashboard |
| **NetworkX** | Latest | Graph analysis |
| **Matplotlib/Seaborn** | Latest | Visualization |

## 🎮 Usage

### 1. Exploratory Data Analysis

Start by analyzing the dataset structure and characteristics:

```bash
# Generate comprehensive EDA report
python helpers/data_viewer.py

# Validate dataset schema
python helpers/schema_viewer.py
```

This generates detailed reports in `out/eda_out/` including:
- Dataset statistics and distributions
- Missing data analysis  
- Attack type distributions
- Message frequency patterns
- Arbitration ID statistics

### 2. Graph Convolutional Network

Train a GCN for anomaly detection:

```bash
python src/GCNN.py
```

**Features:**
- Converts CAN messages to graph representations
- Learns node embeddings for Arbitration IDs
- Builds correlation-based adjacency matrices
- Generates anomaly scores for each message
- Creates visualizations: PCA plots, score distributions, graph structures

**Outputs:**
- `outputs/X.npy`: Node feature matrix
- `outputs/edge_index.npy`: Graph adjacency matrix  
- `outputs/node_embeddings_cpu.npy`: Learned embeddings
- `outputs/node_anomaly_score_cpu.npy`: Anomaly scores
- Visualization plots (PCA, histograms, graph structures)
  Currently a Work in progress

### 3. Traditional ML Pipeline

Run baseline and ensemble models:

```bash
python src/ML.py
```

**Models:**
- Random Forest classifier with feature engineering
- XGBoost classifier with hyperparameter optimization
- Comprehensive feature extraction pipeline
- Cross-validation and performance metrics

### 4. Hybrid ML Framework

Train the advanced hybrid model:

```bash
python src/ensemble_trial.py
```

**Architecture Components:**
- **Sequence Transformer**: Captures temporal patterns in message sequences
- **Graph Neural Network**: Models network topology and message relationships  
- **Contrastive Learning**: Learns robust message representations
- **Fusion Classifier**: Combines all modalities for final predictions

**Features:**
- Sliding window approach for sequence modeling
- Multi-modal feature fusion
- PyTorch AMP for efficient training
- Comprehensive evaluation metrics


### 5. Interactive Dashboard

Launch the Streamlit application:

```bash
streamlit run utils/visual.py
```

**Dashboard Features:**
- Interactive data upload and preprocessing
- Real-time model training and evaluation
- Confusion matrix visualization
- Feature importance analysis
- Performance comparison charts

## 🔬 Technical Methodology

### Graph Neural Network Approach

The GCN implementation treats CAN messages as nodes in a graph where:

1. **Node Features**: 
   - Arbitration ID embeddings
   - Payload byte statistics (mean, frequency)
   - Message timing characteristics

2. **Edge Construction**:
   - Correlation-based adjacency matrix
   - Top-k neighborhood selection
   - Threshold-based edge pruning

3. **Architecture**:
   - 2-layer Graph Convolutional Network
   - Reconstruction loss for unsupervised learning
   - Anomaly scoring through embedding distances

### Hybrid Framework

The hybrid approach combines multiple modalities:

1. **Sequence Component**:
   - Transformer encoder for temporal patterns
   - Multi-head attention mechanism
   - Positional encoding for message sequences

2. **Graph Component**:
   - GCN layers for network topology
   - Global mean pooling for graph-level features
   - Message relationship modeling

3. **Contrastive Component**:
   - Self-supervised representation learning
   - Message similarity modeling
   - Robust feature extraction

4. **Fusion Strategy**:
   - Multi-modal feature concatenation
   - Dropout for regularization
   - Binary classification head

### Feature Engineering

Comprehensive feature extraction pipeline:

- **Payload Features**: Byte-level analysis, entropy calculation, statistical moments
- **Timing Features**: Inter-arrival times, frequency estimation, burst detection
- **Network Features**: Message frequency per ID, traffic patterns
- **Statistical Features**: Mean, standard deviation, correlations, distributions

<!-- ## �� Results & Performance

### Dataset Characteristics
- **Total Messages**: 8,694,507
- **Attack Ratio**: ~10% (873,242 attack messages)
- **Message Types**: 4 attack categories + normal traffic
- **Temporal Span**: Real vehicle communication sessions

### Model Performance
- **GCN Anomaly Detection**: Unsupervised learning with reconstruction loss
- **Hybrid Framework**: Multi-modal fusion with attention mechanisms
- **Traditional ML**: Ensemble methods with feature engineering
- **Real-time Processing**: Streamlit dashboard for interactive analysis

### Output Artifacts
- **Model Checkpoints**: Trained model weights and configurations
- **Embeddings**: Learned representations for visualization
- **Evaluation Metrics**: Classification reports, confusion matrices, ROC curves
- **Visualizations**: PCA plots, graph structures, performance charts -->

## 🛠️ Development

### Adding New Models

1. **Create Model File**: Add new implementation in `src/` directory
2. **Follow Patterns**: Use existing data loading and preprocessing utilities
3. **Add Evaluation**: Include comprehensive metrics and visualizations
4. **Update Documentation**: Document new approaches and results

### Extending Data Processing

1. **Custom EDA**: Modify `helpers/data_viewer.py` for specialized analysis
2. **Preprocessing**: Update functions in model files or `MISC/preprocess.py`
3. **Feature Engineering**: Add new feature extraction methods
4. **Validation**: Use `helpers/schema_viewer.py` for data quality checks

### Enhancing Visualizations

1. **Dashboard**: Extend `utils/visual.py` with new Streamlit components
2. **Plotting**: Add model-specific visualization functions
3. **Real-time**: Implement live monitoring capabilities
4. **Export**: Add report generation and export functionality

## 📚 Research & References

### Dataset
- [Car Hacking: Attack & Defense Challenge 2020](https://ieee-dataport.org/open-access/car-hacking-attack-defense-challenge-2020-dataset)
- [IEEE Dataport](https://ieee-dataport.org/)

### Key Papers
- Graph Neural Networks for CAN Bus Intrusion Detection
- Transformer-based Sequence Modeling for Automotive Security
- Multi-modal Fusion for Vehicle Cybersecurity

### Libraries & Frameworks
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
- [Streamlit](https://docs.streamlit.io/) - Interactive dashboards
- [Polars](https://pola.rs/) - Fast data processing
- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes
- Ensure backward compatibility

<!-- ## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## ⚠️ Disclaimer

This project is developed for **educational and research purposes** only. Always ensure compliance with local regulations and ethical guidelines when working with automotive systems and cybersecurity research.

## 🤝🙌 Acknowledgments

- **Amrita Vishwa Vidyapeetham** IEEE Student Branch on organising the hackathon.
- **IEEE Dataport** for providing the Car Hacking dataset
- **PyTorch Community** for excellent deep learning frameworks
- **Automotive Security Research Community** for ongoing contributions
- **Open Source Contributors** who make projects like this possible

---

## 👥 Team Members

| Name           | GitHub                                | LinkedIn                                        |
|----------------|---------------------------------------|------------------------------------------------|
| Aryan jaljith  | [GitHub](https://github.com/aryanjaljith04) | [LinkedIn](https://www.linkedin.com/in/aryan-jaljith-64283b240/) |
| Mauli Rajguru  | [GitHub](https://github.com/maulirajguru) | [LinkedIn](https://www.linkedin.com/in/maulir/) |
| Anmol  | [GitHub](https://github.com/Anmol-G-K) | [LinkedIn](https://www.linkedin.com/in/anmolkrish/) |
---
<div align="center">

**🔒 Securing the Future of Connected Vehicles 🔒**

*Advanced Machine Learning for Automotive Cybersecurity*

</div>