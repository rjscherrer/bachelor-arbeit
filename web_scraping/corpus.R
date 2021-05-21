##############################################
## CORPUS
##############################################
## this script is used to create the corpus containing all scraped data.
## data is stored in xml format. every data category gets its own folder.

create_corpus_category <- function(corpus_category, article_category, create_article_category=TRUE) {
  ## this function creates a new category in the corpus
  
  # create folders to separate between term of obama and trump
  dir.create(file.path("../corpus", corpus_category))
  dir.create(file.path(paste0("../corpus/", corpus_category), "obama"))
  dir.create(file.path(paste0("../corpus/", corpus_category), "trump"))
  
  if(create_article_category) {
    # create folders for data of departments
    dir.create(file.path(paste("../corpus", corpus_category, "obama", sep = "/"), article_category))
    dir.create(file.path(paste("../corpus", corpus_category, "trump", sep = "/"), article_category))
    
    # create folders for data of secretaries
    dir.create(file.path(paste("../corpus", corpus_category, "obama", sep = "/"), "speeches"))
    dir.create(file.path(paste("../corpus", corpus_category, "trump", sep = "/"), "speeches"))
  }
}

add_article_to_corpus <- function(corpus_category, article_category, file_name, meta_data, paragraphs, no_article_category = FALSE) {
  ## this function adds a scraped article to the corpus
  
  doc <- newXMLDoc()
  
  # create xml node containing article data
  article <- newXMLNode("article", doc = doc)
    newXMLNode("date", format(meta_data$date, "%d.%m.%Y"), parent = article)
    newXMLNode("title", meta_data$title, parent = article)
    newXMLNode("url", meta_data$url, parent = article)
    newXMLNode("scrape_date", format(meta_data$scrape_date, "%d.%m.%Y"), parent = article)
    newXMLNode("president", meta_data$president, parent = article)
    if(!no_article_category) {
      newXMLNode("category", article_category, parent = article)
    }
      
    text <- newXMLNode("text", parent = article)
      for (paragraph in paragraphs) {
        newXMLNode("paragraph", paragraph, parent = text)
      }
    
  # save xml file to corpus
  if(!no_article_category) {
    saveXML(doc = doc, file = paste("../corpus", corpus_category, article_category, paste0(file_name, ".xml"), sep = "/"))
  } else {
    saveXML(doc = doc, file = paste("../corpus", corpus_category, paste0(file_name, ".xml"), sep = "/"))
  }
}
