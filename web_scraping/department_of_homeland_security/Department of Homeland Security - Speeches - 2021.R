# scraper for the department of homeland security
source("config.R")
source("corpus.R")

# scraper settings
corpus_category <- "department_of_homeland_security"
article_category <- "speeches"
page_version <- "2021"
main_url <- "https://www.dhs.gov/archive/news-releases/speeches?combine=&year_filter%5Bvalue%5D%5Byear%5D=&month_filter%5Bvalue%5D%5Bmonth%5D=&month_filter%5Bvalue%5D%5Byear%5D=&field_taxonomy_topics_tid=All&items_per_page=25&page="
start_page <- 0
end_page <- 5

# function to get all articles
get_articles <- function(html) {
  title <- xpathSApply(html, "//*[@id='content-area']/div[2]/div/div[2]/div[2]/div/div/span/a", xmlValue)
  date <- xpathSApply(html, "//*[@id='content-area']/div[2]/div/div[2]/div[2]/div/span/span", xmlValue) %>%
    as.Date(format = "%B %d, %Y", origin = "1970-01-01")
  url <- paste0("https://www.dhs.gov", xpathSApply(html, "//*[@id='content-area']/div[2]/div/div[2]/div[2]/div/div/span/a", xmlGetAttr, "href"))
  id <- url
  
  articles <- data.frame(id, title, date, url, stringsAsFactors = F)
  names(articles) <- c("id", "title", "date", "url")
  
  return(articles)
}

# create article df
articles <- data.frame(id=character(),
                       title=character(),
                       date=as.Date(character()),
                       url=character(),
                       stringsAsFactors=FALSE)

# main scrape
pb_articles <- txtProgressBar(min = start_page, max = end_page, initial = start_page, char = "#", style = 3)
for(page_index in (start_page:end_page)) {
  html <- htmlParse(GET(paste0(main_url, page_index)), asText=T, encoding = "UTF-8")
  articles <- rbind(articles, get_articles(html = html))
  setTxtProgressBar(pb_articles, page_index)
}
close(pb_articles)

# add further information and save data to disk
articles$scrape_date <- Sys.Date()
articles <- articles %>% filter(
  date >= start_date & date <= end_date
)
articles$president <- ifelse(articles$date < cut_date, "obama", "trump")
save(articles, file = paste0("./", corpus_category, "/", "articles - ", article_category, " - ", page_version, ".RData"))

# function to get the text of an article
get_article_text <- function(html, xpath_body) {
  text <- xpathSApply(html, paste0(xpath_body, "//text()"), xmlValue) %>%
    trimws() %>%
    str_replace_all("  +", " ") %>%
    .[nchar(.) > 10]
  
  if(length(text) > 0) {
    return(list("text" = text, "error" = NA))
  } else {
    return(list("text" = NA, "error" = "text not found"))
  }
}

# create corpus category
create_corpus_category(corpus_category = corpus_category, 
                       article_category = article_category)

# prepare to scrape article meta data
scrape_attempts <- 0
failed_scrapes <- data.frame(row_nr=numeric(),
                             url=character(), 
                             error_code=character(),
                             stringsAsFactors=FALSE)

# add meta data and save article to disk
pb_meta <- txtProgressBar(min = 0, max = nrow(articles), initial = 0, char = "#", style = 3)
for(i in 1:nrow(articles)) {
  setTxtProgressBar(pb_meta, i)
  scrape_attempts <- scrape_attempts + 1
  html <- htmlParse(GET(url = articles[i,]$url), asText=T, encoding = "UTF-8")
  xpath_body = "//*[@id='content-area']/div/div/article/div[1]/div[contains(@class, 'field')]"
  
  # get the text of the article
  article_text <- get_article_text(html = html, xpath_body = xpath_body)
  if(!is.na(article_text$error)) {
    failed_scrapes <- failed_scrapes %>% add_row(row_nr = i, url = articles[i,]$url, error_code = article_text$error)
    next
  }
  
  # save article to disk
  add_article_to_corpus(corpus_category = paste(corpus_category, articles[i,]$president, sep = "/"),
                        article_category = article_category,
                        file_name = paste0(digest(articles[i, !names(articles) %in% "scrape_date"], algo="md5", serialize=T), "-", page_version),
                        meta_data = articles[i,],
                        paragraphs = article_text$text)
}
close(pb_meta)

# scraping done
save(failed_scrapes, file = paste0("./", corpus_category, "/", 
                                   round((nrow(failed_scrapes)/scrape_attempts)*100, digits = 2), 
                                   "% failed scrapes (", 
                                   nrow(failed_scrapes),
                                   " out of ",
                                   scrape_attempts,
                                   ") - ",
                                   article_category,
                                   " - ",
                                   page_version, 
                                   ".RData"))

print("Scrape complete.")
