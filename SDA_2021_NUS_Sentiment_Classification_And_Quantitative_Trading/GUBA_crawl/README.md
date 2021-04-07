[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **GUBA_crawl** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of Quantlet: 'GUBA_crawl'

Published in: 'SDA_2021_NUS'

Description: 'Project_Group8 file'

Keywords: 'NUS, FE5225, tieba, stock comment, crawl'

Author: 'Wang Yuzhou'

```

### PYTHON Code
```python

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import os
os.chdir(r'C:/Users/ASUS/Desktop/teiba/')

if __name__ == '__main__':
    process = CrawlerProcess(get_project_settings())
    process.crawl('single_stock')    #  你需要将此处的spider_name替换为你自己的爬虫名称
    process.start()
```

automatically created on 2021-04-01