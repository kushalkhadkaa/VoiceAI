import urllib.request
import urllib.parse
from html.parser import HTMLParser

class DDGParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self.in_title = False
        self.in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        
        if tag == "div" and "web-result" in cls:
            if self.current_result and self.current_result.get("url"):
                self.results.append(self.current_result)
            self.current_result = {"title": "", "url": "", "snippet": ""}
            
        elif self.current_result is not None:
            if tag == "a" and "result__a" in cls:
                self.in_title = True
                self.current_result["url"] = attrs_dict.get("href", "")
            elif tag == "a" and "result__snippet" in cls:
                self.in_snippet = True
            elif tag == "a" and "result__url" in cls:
                self.current_result["url"] = attrs_dict.get("href", "")

    def handle_endtag(self, tag):
        if tag == "a":
            self.in_title = False
            self.in_snippet = False

    def handle_data(self, data):
        if self.current_result is not None:
            if self.in_title:
                self.current_result["title"] += data
            elif self.in_snippet:
                self.current_result["snippet"] += data

    def close(self):
        super().close()
        if self.current_result and self.current_result.get("url"):
            self.results.append(self.current_result)
        # Clean URLs
        for r in self.results:
            url = r["url"]
            if "uddg=" in url:
                parsed = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed.query)
                if "uddg" in qs:
                    url = qs["uddg"][0]
            r["url"] = url

# Run query
query = "nepal"
url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8")
        parser = DDGParser()
        parser.feed(html)
        parser.close()
        print("FOUND RESULTS:", len(parser.results))
        for r in parser.results[:5]:
            print("---")
            print("TITLE:", r["title"].strip())
            print("URL:", r["url"].strip())
            print("SNIPPET:", r["snippet"].strip())
except Exception as e:
    print("ERROR:", e)
