# scraper for the department of sate
source("config.R")
source("corpus.R")

# scraper settings
corpus_category <- "department_of_state"
article_category <- "remarks_kerry"
page_version <- "2017"
main_url <- c("https://2009-2017.state.gov/secretary/remarks/", "/index.htm")
start_page <- 2013
end_page <- 2017

# function to get all articles
get_articles <- function(html) {
  title <- xpathSApply(html, "//*[@id='main-content']/div[3]/div/div[2]/div/div/a", xmlValue)
  date <- xpathSApply(html, "//*[@id='main-content']/div[3]/div/div[2]/div/div/strong/span", xmlValue) %>%
    as.Date(format = "%m/%d/%y", origin = "1970-01-01")
  url <- paste0("https://2009-2017.state.gov", xpathSApply(html, "//*[@id='main-content']/div[3]/div/div[2]/div/div/a", xmlGetAttr, "href"))
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
  html <- htmlParse(httr::content(GET(paste0(main_url[1], page_index, main_url[2])), as = "text", encoding = "UTF-8"), 
                    asText=T, encoding = "UTF-8")
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
  html <- htmlParse(httr::content(GET(url = articles[i,]$url), as = "text", encoding = "UTF-8"), 
                    asText=T, encoding = "UTF-8")
  xpath_body = "//*[@id='centerblock']"
  
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
