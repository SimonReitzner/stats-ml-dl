import lifelines.datasets
import yfinance as yf

class Data():

    def get_kidney_transplant(self, file: str="kidney_transplant.csv"):
        df =  lifelines.datasets.load_kidney_transplant()
        df.to_csv(
            file,
            header=True,
            sep=",",
            index=False
        )
    
    def get_stock_prices(self, file: str="stock_prices.csv"):
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="max")
        df = df.reset_index(drop=False)
        df.to_csv(
            file,
            header=True,
            sep=",",
            index=False
        )


if __name__ == "__main__":
    data = Data()
    data.get_kidney_transplant()
    data.get_stock_prices()