SELECT *FROM fraud_detection_project.credit_card_fraud_2026;
SELECT COUNT(*) AS total_transactions
FROM credit_card_fraud_2026;
DESCRIBE fraud_detection_project.credit_card_fraud_2026;
SELECT *
FROM fraud_detection_project.credit_card_fraud_2026
LIMIT 10;
SELECT COUNT(*) AS total_transactions
FROM fraud_detection_project.credit_card_fraud_2026;
DESCRIBE fraud_detection_project.credit_card_fraud_2026;
SELECT ROUND(SUM(amount_usd), 2) AS total_transaction_amount
FROM fraud_detection_project.credit_card_fraud_2026;
SELECT ROUND(AVG(amount_usd), 2) AS average_transaction_amount
FROM fraud_detection_project.credit_card_fraud_2026;
SELECT *
FROM fraud_detection_project.credit_card_fraud_2026
ORDER BY amount_usd DESC
LIMIT 10;
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'fraud_detection_project'
  AND TABLE_NAME = 'credit_card_fraud_2026'
ORDER BY ORDINAL_POSITION;