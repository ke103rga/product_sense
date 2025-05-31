# Product sense

This library is designed for storing, preprocessing, and analyzing user event data.  
It features clustering methods, metrics evaluation, and tools for product funnel and cohort analysis.  

You can use this library to study user behavior, segment users, and form hypotheses about what motivates users to take desired actions or abandon a product.
Product sense uses clickstream data to build behavioral segments, highlighting events and patterns in user behavior that affect your conversion rates, retention, and revenue. The library is built for data analysts, marketers, product owners, managers, and anyone else whose job it is to improve product quality.

As a natural part of the Jupyter environment, it extends the capabilities of the pandas, plotly, and scikit-learn libraries to more efficiently process sequential event data. The customer retention tools are interactive and designed for analytics, so you don't have to be a Python expert to use them. With just a few lines of code, you can process data, explore customer journey maps, and create visualizations.


# Getting Started with Product Sense

This guide provides instructions on how to install the `product_sense` library and includes a few basic usage examples to get you started.

## Installation

Currently, `product_sense` can be installed directly from the GitHub repository using pip:

```bash
pip install --upgrade git+https://github.com/ke103rga/product_sense.git
```

This command will:

- Install pip if it’s not already installed.
- Upgrade pip to the latest version.
- Install the product_sense package directly from the specified Git repository.

Note: Make sure you have Git installed on your system for this command to work correctly. You can download Git from https://git-scm.com/downloads.

## Basic Usage Examples
<hr>

Here are a few basic examples demonstrating how to use the product_sense library.

### 1. Creating and Preprocessing an EventFrame
This example shows how to create an EventFrame object and split a dataset into sessions based on user inactivity.

```python
from product_sense.eventframing.eventframe import EventFrame
from product_sense.eventframing.preprocessors import SplitSessionsPreprocessor
import pandas as pd

# Sample data (replace with your actual data)
data = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 2],
    'event': ['page_view', 'button_click', 'page_view', 'page_view', 'form_submit', 'logout'],
    'timestamp': ['2023-10-26 10:00:00', '2023-10-26 10:02:00', '2023-10-26 10:10:00',
                  '2023-10-26 11:00:00', '2023-10-26 11:05:00', '2023-10-26 11:10:00']
})

# Define the column schema
cols_schema = {
    'user_id': 'user_id',
    'event_name': 'event',
    'event_timestamp': 'timestamp'
}

# Create an EventFrame object
ef = EventFrame(data, cols_schema=cols_schema, prepare=True)

# Split the dataset into sessions
# A new session starts after 15 minutes of inactivity
split_sessions_preprocessor = SplitSessionsPreprocessor(timeout=(15, 'm'))
ef = split_sessions_preprocessor.apply(ef)

ef.to_dataframe() # Display the resulting EventFrame (with session IDs)
```

Explanation:

- First, necessary modules are imported.
- Sample data is created using a pandas DataFrame, replace it with yours.
- An `EventFrame` object is created, specifying the column schema and enabling data preparation.
- A `SplitSessionsPreprocessor` is used to split the `EventFrame` into sessions based on a 15-minute timeout.

### 2. Analyzing Descriptive Statistics
This example demonstrates how to analyze descriptive statistics of an EventFrame, including session-level statistics

```python
from product_sense.ux_researching import DescStatsAnalyzer

# Create a DescStatsAnalyzer object
desc_stats_analyzer = DescStatsAnalyzer()

# Describe the EventFrame
results = desc_stats_analyzer.describe(
    ef,
    add_path_stats=False,
    add_session_stats=True
)

results
```

Explanation:


- A DescStatsAnalyzer object is created.
- The describe method is called on the EventFrame to calculate descriptive statistics, including session statistics. Path statistics are disabled in this example.

# Practice Examples

This section provides links to comprehensive examples demonstrating how to use the `product_sense` library effectively.  These examples showcase the power and flexibility of the library for various product analytics and UX research tasks.


1.  **Basic Usage of Key Functions:**
    *   [Link to basic usage example](INSERT_LINK_TO_BASIC_EXAMPLE_HERE) - This example provides a step-by-step guide to using all the core functions of the `product_sense` library. It covers data loading, preprocessing, event framing, sessionization, and basic statistical analysis. It's a great starting point for understanding the fundamental capabilities of the library.

2.  **Full-Scale Research Project:**
    *   [Link to full-scale research project](INSERT_LINK_TO_FULL_RESEARCH_PROJECT_HERE) - This example demonstrates how to conduct a complete UX research project using `product_sense`. It includes data collection, hypothesis formulation, in-depth data analysis, and result interpretation. This example illustrates how the library can be applied to real-world research scenarios to gain valuable insights into user behavior and product performance.