# Import scrapy
import scrapy
import time
import pandas as pd
import service_identity

# Import the CrawlerProcess
from scrapy.crawler import CrawlerProcess


url = 'file:///C:/Users/proto/Desktop/Joel/protoqsar/pipeline/utils/Chemid_toxnet/output_5_3.xml'

ENDPOINT_INFO = {
    'organism': 'rat',
    'test-type': 'LC50',
    'route': 'inhalation',
}


# Create the Spider class
class ToxnetSpider(scrapy.Spider):
    name = 'toxnetspider'
    custom_settings = {
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'DOWNLOAD_DELAY': 3.5
    }

    # start_requests method
    def start_requests(self):
        yield scrapy.Request(url = url, callback = self.get_mol_links, priority=1)

    def get_mol_links(self, response):
        cas_list = response.xpath('//CASRegistryNumber/text()').extract()
        # cas_list = response.xpath('//td[@class="innerCol1 bodytext"]/text()').extract()
        for cas in cas_list:
            cas_links.append('https://chem.nlm.nih.gov/chemidplus/number/{}'.format(cas))

        for link in cas_links:
            # The page only allows one automatic request every three seconds
            yield response.follow(url = link, callback = self.get_endpoint, priority=1)

    def get_endpoint(self, response):
        cas = response.url.split('/')[-1]
        print('[PARSED]', cas)

        table_data = response.xpath('//tr[@class="TableRow"]//text()').extract()

        # print(''.join(table_data).split('\xa0'))

        outpath_table = 'F:/web-scraping/chemidplus/all/{}.xurros'.format(cas)
        with open(outpath_table, 'w') as outfile:
            outfile.write(''.join(table_data))

        for n in range(len(table_data)):
            if table_data[n] == ENDPOINT_INFO['organism']:
                if table_data[n+2] == ENDPOINT_INFO['test-type']:
                    if table_data[n+4] == ENDPOINT_INFO['route']:
                        cas_ids.append(cas)
                        s = table_data[n+6]
                        value = s[s.find("(")+1:s.find(")")]
                        endpoint_list.append(value)

                        df = pd.Series({
                            'CAS': str(cas),
                            'organism': ENDPOINT_INFO['organism'],
                            'test-type': ENDPOINT_INFO['test-type'],
                            'route': ENDPOINT_INFO['route'],
                            'value': str(value)
                        }).to_frame().T

                        print('Aleluya! Qui no menja pa menja xulla!')
                        print(df)

                        outpath = 'F:/web-scraping/chemidplus/{}/{}.txt'.format(
                            '_'.join(list(ENDPOINT_INFO.values())),
                            cas
                        )
                        df.to_csv(
                            outpath,
                            index=False,
                            header=True,
                            sep='\t',
                            encoding='utf-8'
                        )


# Initialize the dictionary **outside** of the Spider class
cas_links = []
cas_ids = []
endpoint_list = []
endpoint_dict = dict()
full_list = []

# Run the Spider
process = CrawlerProcess()
process.crawl(ToxnetSpider)
process.start()

# Print results
endpoint_dict = dict(zip(cas_ids, endpoint_list))
print(endpoint_dict)
