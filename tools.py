import pandas as pd


def csv_to_xlsx(csv_file, xlsx_file):
    """
    Transfer data from CSV to XLSX format.
    
    Args:
        csv_file (str): Path to the input CSV file
        xlsx_file (str): Path to the output XLSX file
    """
    df = pd.read_csv(csv_file)
    df.to_excel(xlsx_file, index=False)
    print(f"Successfully converted {csv_file} to {xlsx_file}")

def main():
    csv_file = "data\\SampleSuperstore.csv"  # Replace with your CSV file path
    xlsx_file = "data\\SampleSuperstore.xlsx"  # Desired XLSX file path
    csv_to_xlsx(csv_file, xlsx_file)
if __name__ == "__main__":
    main()