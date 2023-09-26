import scrapy


class MainSpider(scrapy.Spider):
    name = "main"
    allowed_domains = ["turbo.az"]
    start_urls = ["https://turbo.az/autos/7267410-geely-tugella-s"]

    def parse(self, response):
        yield from self.parse_car_details(response)

    def parse_car_details(self, response):
        yield {
            'title'     : response.css('h1.product-title::text').get(),
            'city'      : response.css('label.product-properties__i-name:contains("Şəhər") + span.product-properties__i-value::text').get(),
            'brand'     : response.css('label.product-properties__i-name:contains("Marka") + span.product-properties__i-value a::text').get(),
            'fields'    : response.css('label.product-properties__i-name::text').getall(),
            'values'    : response.css('span.product-properties__i-value::text').getall(),
            'price'     : response.css('div.product-price__i--bold::text').get(),
            'update'    : response.css('span.product-statistics__i-text::text').get(),
            'views'     : response.css('ul.product-statistics li:nth-child(2) span::text').get(),
        }
