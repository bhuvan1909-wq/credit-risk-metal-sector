library(readxl)
library(dplyr)
library(tidyr)
library(caret)
library(pROC)

rating_long <- project1 %>%
  select(Company, `Transition Flag 2022`, `Transition Flag 2023`, `Transition Flag 2024`) %>%
  pivot_longer(cols = starts_with("Transition Flag"),
               names_to = "Year_Flag", values_to = "Transition_Flag") %>%
  mutate(Year = case_when(
    Year_Flag == "Transition Flag 2022" ~ 2022,
    Year_Flag == "Transition Flag 2023" ~ 2023,
    Year_Flag == "Transition Flag 2024" ~ 2024,
    TRUE ~ NA_real_)) %>%
  select(-Year_Flag)
 modeldata <- merge(project, rating_long, by = c("Company", "Year"), all.x = TRUE)

 
 # Replace any NA in Transition_Flag with 0 (assuming no transition if missing)
 modeldata$Transition_Flag[is.na(modeldata$Transition_Flag)] <- 0
 
 # Convert Transition_Flag to factor for classification
 modeldata <- modeldata %>%
   mutate(default_flag = as.factor(Transition_Flag))
 
 # Partitioning data into training and testing sets for model building
 set.seed(123)
 trainIndex <- createDataPartition(modeldata$default_flag, p = 0.7, list = FALSE)
 train <- modeldata[trainIndex, ]
 test <- modeldata[-trainIndex, ]
 
 # logistic regression model
 model <- glm(default_flag ~ `Debt to Equity` + `Interest Coverage` + `Current Ratio` + `ROCE (%)`,
              data = train, family = binomial)
 
 # Summary of model
 summary(model)
 
 # Predict PD (probabilities) for the full dataset, including train and test
 modeldata$pred_prob <- predict(model, newdata = modeldata, type = "response")

 modeldata$pred_label <- as.factor(ifelse(modeldata$pred_prob > 0.5, 1, 0))
 
 
 write.csv(modeldata, "credit_risk_scorecard_predictions_full.csv", row.names = FALSE)
 

 head(modeldata)
 
 
 
 library(ggplot2)
 library(dplyr)
 library(readr)
 

 results <- read.csv("credit_risk_scorecard_predictions_full.csv")
 
 # BASIC SCATTERPLOT: Probability of Default per company for each year
 ggplot(results, aes(x = factor(Year), y = pred_prob, colour = Company)) +
   geom_point(size = 3, alpha = 0.7) +
   geom_line(aes(group = Company), alpha = 0.7) +
   labs(title = "Probability of Default per Company, per Year",
        x = "Year", y = "Predicted PD (Scorecard Model)") +
   theme_minimal()
 
 # BOX PLOT: Distribution of PDs by Year
 ggplot(results, aes(x = factor(Year), y = pred_prob)) +
   geom_boxplot(fill = "steelblue", alpha = 0.5) +
   labs(title = "Distribution of PDs by Year",
        x = "Year", y = "Predicted PD") +
   theme_minimal()
 
 # HEATMAP: Company vs. Year PDs
 ggplot(results, aes(x = factor(Year), y = Company, fill = pred_prob)) +
   geom_tile(color = "white") +
   scale_fill_gradient(low = "white", high = "red") +
   labs(title = "Heatmap of Company PDs Over Time",
        x = "Year", y = "Company", fill = "PD") +
   theme_minimal()
 
 # Highlight Vedanta across years 
 ggplot(filter(results, Company == "Vedanta Ltd"), aes(x = factor(Year), y = pred_prob)) +
   geom_point(color = "red", size = 4) +
   geom_line(group = 1, color = "red", size = 1) +
   labs(title = "Vedanta Ltd: Probability of Default Over Years",
        x = "Year", y = "Predicted PD") +
   ylim(0, 1) +
   theme_minimal()
 
 
 
 
 
 #sensitivity analysis
 

 base_case <- filter(results, Company == "Vedanta Ltd", Year == 2024)
 
 coefs <- c("(Intercept)" = -2.5,  # Example intercept
            "Debt.to.Equity" = 0.4,
            "Interest.Coverage" = -0.7,
            "Current.Ratio" = -1.2,
            "ROCE.." = -3.0)      # Replace with actual names & values
 
 # Sensitivity for Debt to Equity from base-25% to base+25%
 de_base <- base_case$Debt.to.Equity
 de_seq <- seq(de_base * 0.75, de_base * 1.25, length.out = 50)
 pd_de <- plogis(
   coefs["(Intercept)"] +
     coefs["Debt.to.Equity"] * de_seq +
     coefs["Interest.Coverage"] * base_case$Interest.Coverage +
     coefs["Current.Ratio"] * base_case$Current.Ratio +
     coefs["ROCE.."] * base_case$ROCE..)
 
 # Plotting PD sensitivity vs Debt to Equity
 df_de <- data.frame(Debt_to_Equity = de_seq, PD = pd_de)
 ggplot(df_de, aes(x = Debt_to_Equity, y = PD)) +
   geom_line(size = 1.2, color = "red") +
   labs(title = "PD Sensitivity to Debt-to-Equity Ratio (Vedanta Ltd, 2024)",
        x = "Debt to Equity", y = "Predicted PD") +
   theme_minimal()
 
 ic_base <- base_case$Interest.Coverage
 ic_seq <- seq(ic_base * 0.75, ic_base * 1.25, length.out = 50)
 pd_ic <- plogis(
   coefs["(Intercept)"] +
     coefs["Debt.to.Equity"] * base_case$Debt.to.Equity +
     coefs["Interest.Coverage"] * ic_seq +
     coefs["Current.Ratio"] * base_case$Current.Ratio +
     coefs["ROCE.."] * base_case$ROCE..)
 df_ic <- data.frame(Interest_Coverage = ic_seq, PD = pd_ic)
 ggplot(df_ic, aes(x = Interest_Coverage, y = PD)) +
   geom_line(size = 1.2, color = "blue") +
   labs(title = "PD Sensitivity to Interest Coverage (Vedanta Ltd, 2024)",
        x = "Interest Coverage", y = "Predicted PD") +
   theme_minimal()
 
 # Sensitivity for Current Ratio (from base-25% to base+25%)
 cr_base <- base_case$Current.Ratio
 cr_seq <- seq(cr_base * 0.75, cr_base * 1.25, length.out = 50)
 pd_cr <- plogis(
   coefs["(Intercept)"] +
     coefs["Debt.to.Equity"] * base_case$Debt.to.Equity +
     coefs["Interest.Coverage"] * base_case$Interest.Coverage +
     coefs["Current.Ratio"] * cr_seq +
     coefs["ROCE.."] * base_case$ROCE..)
 df_cr <- data.frame(Current_Ratio = cr_seq, PD = pd_cr)
 ggplot(df_cr, aes(x = Current_Ratio, y = PD)) +
   geom_line(size = 1.2, color = "orange") +
   labs(title = "PD Sensitivity to Current Ratio (Vedanta Ltd, 2024)",
        x = "Current Ratio", y = "Predicted PD") +
   theme_minimal()
 
 # Sensitivity for ROCE (from base-25% to base+25%)
 roce_base <- base_case$ROCE..
 roce_seq <- seq(roce_base * 0.75, roce_base * 1.25, length.out = 50)
 pd_roce <- plogis(
   coefs["(Intercept)"] +
     coefs["Debt.to.Equity"] * base_case$Debt.to.Equity +
     coefs["Interest.Coverage"] * base_case$Interest.Coverage +
     coefs["Current.Ratio"] * base_case$Current.Ratio +
     coefs["ROCE.."] * roce_seq)
 df_roce <- data.frame(ROCE = roce_seq, PD = pd_roce)
 ggplot(df_roce, aes(x = ROCE, y = PD)) +
   geom_line(size = 1.2, color = "green") +
   labs(title = "PD Sensitivity to ROCE (Vedanta Ltd, 2024)",
        x = "ROCE", y = "Predicted PD") +
   theme_minimal()
 
 
 
 
 