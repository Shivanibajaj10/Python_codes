install.packages("tidyverse")
library(tidyverse)

install.packages("readr")
library(readr)
delivery_times <- read_csv("in_class_4_data (1).csv")
read_csv("class4.csv")

delivery_times <- read.csv("/Users/Shivanibajaj/Downloads/in_class_4_data.csv")

View(delivery_times)


###Literals
#in_class_4_data csv

###Functions
#library    -   Load new functions and variables to use
#read_csv   -   (tidyverse) 
#View       -   view dataframe in a nice setting

###variables
#delivery_times

###operators
#<-       -   Assignment


install.packages("ggplot2")
library("ggplot2")
#Plot 1 Bar grpah of delivery estimate
ggplot(delivery_times) +
  aes(x = delivery_id, y = delivery_time_estimate) +
  geom_bar(stat = "identity") +
  labs(title = "Delivery Time Estimate", x = "Delivery ID", y = "Delivery Time Estimate") +
  theme_linedraw()

#Plot 2
install.packages("plotly")
library(plotly)

plot1 <- ggplot(delivery_times, aes(x = delivery_day, y = delivery_time, group = delivery_location)) +
  geom_line() +
  labs(title = "Delivery Times by Location Over Time",
       x = "Delivery Day",
       y = "Delivery Time",
       color = "Delivery Location") +
  scale_x_date(date_labels = "%b %d") +
  theme_linedraw()
ggplotly(plot1)
