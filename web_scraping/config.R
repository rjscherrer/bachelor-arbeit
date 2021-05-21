##############################################
## CONFIG
##############################################
## this script contains all libraries to scrape data. it is imported by every 
## scraper script.

# install all needed libraries
#install.packages("digest")
#install.packages("dplyr")
#install.packages("httr")
#install.packages("stringr")
#install.packages("tidyverse")
#install.packages("webdriver")
#install.packages("XML")
#install.packages("xml2")

# import all needed libraries
library(digest)
library(dplyr)
library(httr)
library(stringr)
library(tidyverse)
library(webdriver)
library(XML)
library(xml2)

# scraper settings
start_date <- as.Date("21.01.2013", format = "%d.%m.%Y", origin = "1970-01-01")
cut_date <- as.Date("21.01.2017", format = "%d.%m.%Y", origin = "1970-01-01")
end_date <- as.Date("20.01.2021", format = "%d.%m.%Y", origin = "1970-01-01")
