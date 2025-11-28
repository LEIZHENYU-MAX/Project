install.packages("here")
library(here)
data <- read.csv("data/city_day.csv", fileEncoding = "UTF-8-BOM")

head(data)

delhi <- subset(data, City == "Delhi")

delhi_selected <- delhi[, c("PM2.5","PM10","NO","NO2","NOx","NH3",
                            "CO","SO2","O3","AQI","AQI_Bucket")]

colSums(is.na(delhi_selected))

delhi_clean <- na.omit(delhi_selected)

nrow(delhi_clean)
head(delhi_clean)
write.csv(delhi_clean, "data/delhi_clean.csv", row.names = FALSE)
