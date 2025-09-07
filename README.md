# Airline Fare Prediction using Graph Neural Networks

This is a comprehensive machine learning project that predicts airline fares using graph neural networks (GNNs) and temporal modeling techniques. This project leverages flight route data from 1993-2024 to build sophisticated models that can forecast airfare prices based on network topology, temporal patterns, and various flight characteristics.

## Project Overview

This project implements multiple approaches to airline fare prediction:

1. **Graph Attention Networks (GAT)** - Uses attention mechanisms to capture important relationships between airports
2. **Temporal Graph Neural Networks** - Models temporal dependencies in flight data across quarters and years
3. **Edge Aggregation Networks** - Focuses on edge-level features and neighbor aggregation for fare prediction
4. **Traditional Graph Convolutional Networks (GCN)** - Baseline approach using standard graph convolutions

## Dataset

### Main Dataset: `flight_routes.csv`
- **Source**: [OpenFlights](https://openflights.org/data.php)
- **Time Period**: 1993-2024
- **Features**: 
  - Flight routes and airport information
  - Passenger counts and market share data
  - Distance metrics (nautical miles)
  - Carrier information (large and low-cost carriers)
  - Fare data for different market segments
  - Temporal information (year, quarter)

## Project Structure

```
CS224WProject/
├── data/
│   ├── airports.txt          # Airport geolocation data
│   ├── flight_routes.csv     # Main flight dataset (1993-2024)
│   └── reduced.csv          # Filtered dataset for faster processing
├── gat_train.py             # Graph Attention Network implementation
├── temporal_gnn.py          # Temporal GNN with edge-to-node attention
├── edge_agg.py              # Edge aggregation network
├── graph_setup.py           # Graph construction and data preprocessing
├── gcn_setup.py             # Graph Convolutional Network setup
├── filter_csv.py            # Data filtering utilities
├── data_features.py         # Feature extraction utilities
├── graph.py                 # Graph visualization and analysis
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

**note data not included in repo due to size**

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CS224WProject
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data files**:
   Ensure the following files are present in the `data/` directory:
   - `airports.txt`
   - `flight_routes.csv`
   - `reduced.csv`

The project includes comprehensive visualization capabilities:
- **Graph Visualization**: NetworkX-based airport network plots
- **Temporal Snapshots**: Quarterly evolution of the flight network
- **Feature Analysis**: Node and edge feature distributions

## Configuration

Key hyperparameters can be adjusted in each model file:
- **Hidden Dimensions**: Model capacity and complexity
- **Learning Rates**: Training speed and convergence
- **Epochs**: Training duration
- **Time Steps**: Temporal modeling granularity
