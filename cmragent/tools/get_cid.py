from selenium import webdriver
from selenium.webdriver.edge.service import Service
from bs4 import BeautifulSoup
import time
import pubchempy as pcp


def get_compound_cid(identifier):
    print(f"\nSearching for CID of: {identifier}")
    try:
        cids = pcp.get_cids(identifier, namespace='name')
        if not cids:
            cids = pcp.get_cids(identifier, namespace='smiles')
        if not cids:
            cids = pcp.get_cids(identifier, namespace='cas')

        if cids:
            cid = cids[0]
            print(f"Found CID: {cid}")
            return cid
        else:
            print("No CID found using PubChemPy, trying Selenium method...")
    except Exception as e:
        print(f"Error using PubChemPy: {e}")
        print("Switching to Selenium method...")

    driver_path = r".\msedgedriver.exe"
    service = Service(executable_path=driver_path)

    edge_options = webdriver.EdgeOptions()
    edge_options.add_argument('--headless')

    driver = webdriver.Edge(service=service, options=edge_options)

    try:
        url = f'https://pubchem.ncbi.nlm.nih.gov/#query={identifier}%2F&tab=compound'
        driver.get(url)

        time.sleep(5)
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')
        span_elements = soup.find_all('span')
        for span in span_elements:
            content = span.get_text().strip()
            if content.isdigit():
                print(f"Found CID via Selenium: {content}")
                return content

        print("No CID found in the page.")
        return None

    except Exception as e:
        print(f"Error using Selenium: {e}")
        return None

    finally:
        driver.quit()