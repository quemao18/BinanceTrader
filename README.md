# BinanceTrader

BinanceTrader is a Python class that utilizes Deep Neural Networks (DNN) for making predictions and executing demo/real trades on the Binance platform. This project aims to provide a framework for algorithmic trading in the cryptocurrency market.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Deploy on Google Cloud](#deploy-on-google-cloud)
- [How It Works](#how-it-works)
- [Risk Disclaimer](#risk-disclaimer)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

## Features

- Leverages Deep Neural Networks for price predictions
- Supports both demo (testnet) and real trading environments (mainnet)
- Automated trading with customizable parameters
- Real-time data fetching and preprocessing
- Integration with Binance API

## Prerequisites

- Python 3.7+
- Binance account (with API access)
- Basic understanding of cryptocurrency trading and Python programming

## Installation

1. Clone this repository:
```
git clone https://github.com/AlrzA2003/BinanceTrader.git
cd BinanceTrader
```

2. Install the required packages:
```
pip install -r requirements.txt
```

For specific versions to ensure compatibility:
```
pip install -r requirements_specific.txt
```

## Setup

1. Data Preparation:
- Run `Downloades.py` to fetch historical data from Binance.
- Execute `Preprocessing.py` to train and save the model and scaler.

Note: While pre-trained models are provided, it's recommended to train your own for the most up-to-date data.

2. Configuration:
- Copy `.env.example` to `.env` and set your Binance credentials and runtime options.
- Recommended defaults for Cloud Run Jobs:
   - `TRADE_MODE=spot`
   - `TESTNET=false`
   - `SINGLE_RUN=true`
   - `MAX_CYCLES_PER_RUN=1`
   - `ONLY_NEW_CANDLE=true`
- Keep `SKIP_FETCH_CURRENCIES=true` to avoid Binance SAPI currency metadata calls that may fail in some cloud regions.

## Usage

After completing the setup, run the BinanceTrader:
```
python BinanceTrader.py
```
The script will connect to your Binance account and start trading based on the DNN predictions.

## Deploy on Google Cloud

This project is optimized for **Cloud Run Jobs** + **Cloud Scheduler**.
Do not deploy it as a Cloud Run Service because this bot is not an HTTP server.

### Prerequisites

- Install and authenticate `gcloud` CLI.
- A GCP project with billing enabled.
- A local `.env` file with at least:

```dotenv
API_KEY=your_binance_api_key
SECRET=your_binance_secret
TRADING_PAIR=BTCUSDT
TIMEFRAME=15m
TESTNET=false
TRADE_MODE=spot
SINGLE_RUN=true
MAX_CYCLES_PER_RUN=1
SLEEP_BETWEEN_CYCLES_SEC=30
ONLY_NEW_CANDLE=true
SKIP_FETCH_CURRENCIES=true
```

### One-Command Deploy

Use the helper script:

```bash
chmod +x deploy.sh
PROJECT_ID=your-project-id ./deploy.sh
```

Optional overrides:

```bash
PROJECT_ID=your-project-id \
REGION=southamerica-east1 \
SCHEDULE="*/15 * * * *" \
JOB_NAME=binance-trader-job \
SCHEDULER_JOB_NAME=run-bot-job \
RUN_NOW=true \
./deploy.sh
```

What `deploy.sh` does:

1. Enables required APIs (Cloud Build, Run, Scheduler, Secret Manager).
2. Builds and pushes image to `gcr.io/<PROJECT_ID>/binance-trader`.
3. Stores `API_KEY` and `SECRET` in Secret Manager.
4. Creates/updates Cloud Run Job with env vars and secrets.
5. Creates/updates Cloud Scheduler job to trigger the run periodically.

### Useful Commands

Run job now:

```bash
gcloud run jobs execute binance-trader-job --region southamerica-east1
```

List executions:

```bash
gcloud run jobs executions list --job binance-trader-job --region southamerica-east1
```

Read logs for one execution:

```bash
gcloud logging read 'resource.type="cloud_run_job" AND labels."run.googleapis.com/execution_name"="<execution-name>"' --limit 100 --format='value(textPayload)'
```

Run deploy

```bash
PROJECT_ID=your-project-id REGION=southamerica-east1 RUN_NOW=true ./deploy.sh
```

## How It Works

BinanceTrader operates by:

1. Connecting to your Binance account using the provided API keys.
2. Making trades with leverage based on predictions from the Deep Neural Network model.
3. Using all available funds in the futures section for trades.

## Risk Disclaimer

Trading cryptocurrencies, especially with leverage, carries significant financial risk. By using BinanceTrader, you acknowledge and accept these risks. Please note:

- This software is for educational and experimental purposes only.
- Never trade with funds you cannot afford to lose.
- The author is not responsible for any financial losses incurred while using this software.
- Always monitor your trades and be prepared to intervene manually if necessary.

## Resources

This project was developed with the help of the following educational resources:

1. **Data Analysis with Pandas and Python**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/data-analysis-with-pandas/)
   - Description: Comprehensive course on data manipulation and analysis using Pandas.

2. **Python for Data Science and Machine Learning Bootcamp**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/)
   - Description: Extensive course covering various machine learning algorithms and their implementation in Python.

3. **Algorithmic Trading A-Z with Python, Machine Learning & AWS**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/algorithmic-trading-with-python-and-machine-learning/)
   - Description: Comprehensive overview of algorithmic trading, from basic concepts to advanced strategies.

4. **Cryptocurrency Algorithmic Trading with Python and Binance**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/cryptocurrency-algorithmic-trading-with-python-and-binance/)
   - Description: Focused course on cryptocurrency trading using the Binance API.

5. **Performance Optimization and Risk Management for Trading**
   - Platform: Udemy
   - [Course Link](https://www.udemy.com/course/performance-optimization-and-risk-management-for-trading/)
   - Description: Course on optimizing trading strategies and managing risk in trading systems.

6. **Python for Finance: Mastering Data-Driven Finance**
   - Author: Yves Hilpisch
   - Publisher: O'Reilly Media
   - [Book Link](https://www.oreilly.com/library/view/python-for-finance/9781492024323/)
   - Description: Comprehensive book covering various aspects of financial analysis and algorithmic trading using Python.

## Contributing

Contributions to improve BinanceTrader are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

