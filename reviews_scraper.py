"""
Credits for this scraper: https://blog.datahut.co/scraping-amazon-reviews-python-scrapy/
This code comes without any warranty and the repository author is not responsible for its usage.
"""
import scrapy


# Creating a new class to implement Spider
class AmazonReviewsSpider(scrapy.Spider):
    # Spider name
    name = 'amazon_reviews'

    # Domain names to scrape
    allowed_domains = ['amazon.it']

    # Base URL for the product
    myBaseUrl = "https://www.amazon.it/Mascherine-Traspiranti-Confortevoli-Anti-polvere-Confezione/product-reviews/B0868J3LB5/"
    start_urls = []

    # Creating list of urls to be scraped by appending page number a the end of base url
    for i in range(1, 121):
        start_urls.append(myBaseUrl + str(i))

    # Defining a Scrapy parser
    def parse(self, response):
        data = response.css('#cm_cr-review_list')

        # Collecting product star ratings
        star_rating = data.css('.review-rating')

        # Collecting user reviews
        comments = data.css('.review-text')
        count = 0

        # Combining the results
        for review in star_rating:
            yield {'stars': ''.join(review.xpath('.//text()').extract()),
                   'comment': ''.join(comments[count].xpath(".//text()").extract())
                   }
            count = count + 1
