# 🚀 Streamlit Deployment Guide

## Quick Start

### Option 1: Run Locally (Easiest)
```bash
# Install dependencies
pip install -r requirements_web.txt

# Run the app
python run_streamlit.py
# OR
streamlit run app.py
```

### Option 2: Deploy on Streamlit Cloud (Free)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Repository: `your-username/your-repo-name`
   - Main file path: `app.py`
   - Requirements file: `requirements_web.txt`
   - Click "Deploy"

## File Structure for Deployment

```
your-repo/
├── app.py                    # Main Streamlit app
├── requirements_web.txt      # Dependencies
├── src/                     # Your source code
│   ├── data_collector.py
│   ├── feature_engineering.py
│   ├── model_trainer_simple.py
│   ├── backtester.py
│   └── utils.py
└── README.md
```

## Configuration

### Environment Variables (Optional)
Create a `.streamlit/secrets.toml` file for any API keys:
```toml
[api_keys]
# Add any API keys here if needed
```

### Customizing the App

You can modify `app.py` to:
- Change the page title and icon
- Add more stock symbols
- Customize the UI layout
- Add more model types

## Troubleshooting

### Common Issues:

1. **Import Errors:**
   - Make sure all files in `src/` are included in your repository
   - Check that `requirements_web.txt` has all dependencies

2. **Data Collection Issues:**
   - Some stocks might not have data available
   - Try different time periods or symbols

3. **Model Training Issues:**
   - Ensure you have enough data (at least 100 records)
   - Try different stock symbols

### Local Development:

```bash
# Install in development mode
pip install -e .

# Run with debug info
streamlit run app.py --logger.level debug
```

## Features of Your App

✅ **Stock Data Collection** - Downloads from Yahoo Finance  
✅ **Technical Analysis** - 80+ indicators  
✅ **Machine Learning** - Random Forest model  
✅ **Backtesting** - Trading simulation  
✅ **Interactive Charts** - Plotly visualizations  
✅ **Performance Metrics** - Comprehensive analysis  

## Next Steps

1. **Add More Models:**
   - LSTM neural networks
   - XGBoost
   - Ensemble methods

2. **Enhance UI:**
   - Add more interactive features
   - Include portfolio management
   - Real-time data updates

3. **Scale Up:**
   - Add user authentication
   - Database integration
   - Email notifications

## Support

- Streamlit Documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Community Forum: [discuss.streamlit.io](https://discuss.streamlit.io)
- GitHub Issues: [github.com/streamlit/streamlit](https://github.com/streamlit/streamlit)

---

**Your app is ready to deploy! 🎉** 