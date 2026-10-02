import yfinance as yf

pairs = {
    "USDJPY": "JPY=X",
    "USDSGD": "SGD=X",
    "USDKRW": "KRW=X",
    "USDINR": "INR=X",
    "AUDUSD": "AUDUSD=X",
    "USDTHB": "THB=X",
    "USDPHP": "PHP=X",
    "USDIDR": "IDR=X",
    "USDTWD": "TWD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCNH": "CNH=X",
}

for pair, ticker in pairs.items():
    print("=" * 60)
    print(f"Testing {pair} ({ticker})")

    try:
        data = yf.download(
            ticker,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print("NO DATA")
            continue

        print(f"Rows: {len(data)}")
        print(f"From: {data.index.min().date()}")
        print(f"To:   {data.index.max().date()}")

        print("\nLatest 3 closes:")
        print(data.tail(3)[["Close"]])

        # Simple usability check
        if len(data) >= 240:
            print("\nSTATUS: GOOD - enough history for ~1 year VaR")
        elif len(data) >= 100:
            print("\nSTATUS: PARTIAL - usable, but less than ideal")
        else:
            print("\nSTATUS: BAD - insufficient historical data")

    except Exception as e:
        print(f"ERROR: {e}")

    print()