CREATE DATABASE analytics_db;
USE analytics_db;

-- Criar a tabela customers_churn
CREATE TABLE customers_churn (
    customerID VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    SeniorCitizen TINYINT,             -- 0 = não, 1 = sim
    Partner VARCHAR(3),                -- Yes / No
    Dependents VARCHAR(3),             -- Yes / No
    tenure INT,                        -- Meses de contrato
    PhoneService VARCHAR(3),           -- Yes / No
    MultipleLines VARCHAR(20),
    InternetService VARCHAR(20),       -- DSL, Fiber optic, None
    OnlineSecurity VARCHAR(20),
    OnlineBackup VARCHAR(20),
    DeviceProtection VARCHAR(20),
    TechSupport VARCHAR(20),
    StreamingTV VARCHAR(20),
    StreamingMovies VARCHAR(20),
    Contract VARCHAR(20),              -- Month-to-month, One year, Two year
    PaperlessBilling VARCHAR(3),       -- Yes / No
    PaymentMethod VARCHAR(50),         -- Ex: Electronic check, Credit card
    MonthlyCharges DECIMAL(10,2),
    TotalCharges DECIMAL(10,2),
    Churn VARCHAR(3)                   -- Yes / No
);

select
	gender,
	count(*) as "Total"
from customers_churn
group by gender;

select * from churn_predictions;

DELETE t1
FROM churn_predictions t1
JOIN churn_predictions t2
  ON t1.customerID = t2.customerID
 AND t1.last_prediction_date < t2.last_prediction_date;
 
 DESCRIBE churn_predictions;
 
 drop table churn_predictions;
 
 CREATE TABLE IF NOT EXISTS churn_predictions (
  customerID VARCHAR(50) NOT NULL,   
  MonthlyCharges DECIMAL(10,2) NULL,  
  tenure INT NULL, 
  churn_probability DECIMAL(6,5) NOT NULL, 
  last_prediction_date DATE NOT NULL,  
  PRIMARY KEY (customerID)                    
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;