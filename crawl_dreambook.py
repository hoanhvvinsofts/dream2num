from bs4 import BeautifulSoup
import requests
import csv


def crawl_page(csvwriter, url="https://xskt.com.vn/so-mo/"):
    global count
    req = requests.get(url)
    soup = BeautifulSoup(req.text, "lxml")

    soup = soup.find("tbody")
    for row in soup.find_all("tr"):
        row = row.find_all("td")
        if row != []:
            try:
                count, word, numbers = row[0].text, row[1].text, row[2].text
                print(count, word, numbers)
            except IndexError:
                print(">> Exception, row = ", row)
            
            csvwriter.writerow([str(count), str(word), str(numbers)])

def main():
    filename = "dreambook.csv"
    with open(filename, 'w', encoding="utf-8", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["STT", "Word", "Lucky_number"])
        for i in range(1, 10):
            print(i)
            crawl_page(url=f"https://xskt.com.vn/so-mo/?pg={i}", csvwriter=csvwriter)

def preprocessing_csv(new_file="dreambook_preprocessed.csv"):
    with open(new_file, 'w', encoding="utf-8", newline='') as csvfile:
        with open("dreambook.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                new_line = line[0].lower().split(",")
                if len(new_line) > 3:
                    new_line[2] = "-".join([elem.replace(" ", "") for elem in new_line[2:len(new_line)]])
                    new_line = new_line[0:3]
                    
                new_line[2] = new_line[2].replace(' ', '').replace('"', '')
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(new_line)


# main()
# preprocessing_csv()
