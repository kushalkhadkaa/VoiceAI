"""Create kyc_db database with customer_kyc table/view and seed or migrate data.

Note: the target server is MariaDB 10.4, which rejects non-deterministic
functions (CURDATE) in generated columns. The `age` column is therefore
created as a VIRTUAL column is NOT possible — instead we expose `age`
through the `customer_kyc_v` view so NL->SQL queries can still use it.
All other generated columns are deterministic and kept as-is.

If a legacy `voiceai.customer_kyc` table exists, rows are copied into
`kyc_db.customer_kyc` with INSERT IGNORE. The application reads KYC data only
from `kyc_db` (the single consolidated database).
"""
from __future__ import annotations

import mysql.connector

TARGET_DB = "kyc_db"
LEGACY_DB = "voiceai"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_kyc (
    customer_id VARCHAR(50) PRIMARY KEY,
    account_number VARCHAR(30) UNIQUE NOT NULL,
    customer_type ENUM('Individual', 'Corporate') NOT NULL,
    customer_name VARCHAR(150) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    middle_name VARCHAR(50),
    last_name VARCHAR(50) NOT NULL,
    nepali_name VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
    gender ENUM('Male', 'Female', 'Other') NOT NULL,
    date_of_birth DATE NOT NULL,
    marital_status ENUM('Single', 'Married', 'Divorced', 'Widowed') NOT NULL,
    nationality VARCHAR(50) DEFAULT 'Nepali',
    resident_status ENUM('Resident', 'NRN', 'Foreigner') DEFAULT 'Resident',
    citizenship_number VARCHAR(50) UNIQUE,
    citizenship_issue_date DATE,
    citizenship_issue_district VARCHAR(50),
    passport_number VARCHAR(50),
    passport_expiry DATE,
    voter_id VARCHAR(50),
    pan_number VARCHAR(20),
    national_id_number VARCHAR(50),

    mobile_number VARCHAR(15) NOT NULL,
    alternate_mobile VARCHAR(15),
    email_address VARCHAR(100),
    preferred_contact_method ENUM('SMS', 'Email', 'Call') DEFAULT 'SMS',
    home_phone VARCHAR(15),
    emergency_contact_name VARCHAR(100),
    emergency_contact_number VARCHAR(15),
    emergency_relationship VARCHAR(50),

    permanent_province VARCHAR(50),
    permanent_district VARCHAR(50),
    permanent_municipality VARCHAR(100),
    permanent_ward_no INT,
    permanent_address VARCHAR(255),
    current_province VARCHAR(50),
    current_district VARCHAR(50),
    current_municipality VARCHAR(100),
    current_ward_no INT,
    current_address VARCHAR(255),
    address_verification_status ENUM('Verified', 'Unverified') DEFAULT 'Unverified',

    father_name VARCHAR(150),
    mother_name VARCHAR(150),
    grandfather_name VARCHAR(150),
    spouse_name VARCHAR(150),
    number_of_dependents INT DEFAULT 0,
    guardian_name VARCHAR(150),
    nominee_name VARCHAR(150),
    nominee_relation VARCHAR(50),

    employment_status ENUM('Salaried', 'Self-employed', 'Unemployed', 'Retired') NOT NULL,
    occupation VARCHAR(100),
    employer_name VARCHAR(150),
    employer_address VARCHAR(255),
    employer_phone VARCHAR(15),
    designation VARCHAR(100),
    employment_start_date DATE,
    employment_years DECIMAL(4,1),
    business_type VARCHAR(100),
    business_registration_no VARCHAR(50),

    monthly_salary DECIMAL(15,2) DEFAULT 0.00,
    annual_income DECIMAL(15,2) DEFAULT 0.00,
    business_income DECIMAL(15,2) DEFAULT 0.00,
    rental_income DECIMAL(15,2) DEFAULT 0.00,
    remittance_income DECIMAL(15,2) DEFAULT 0.00,
    agricultural_income DECIMAL(15,2) DEFAULT 0.00,
    investment_income DECIMAL(15,2) DEFAULT 0.00,
    other_income DECIMAL(15,2) DEFAULT 0.00,
    total_monthly_income DECIMAL(15,2) GENERATED ALWAYS AS (monthly_salary + business_income + rental_income + remittance_income + agricultural_income + investment_income + other_income),
    total_monthly_expense DECIMAL(15,2) DEFAULT 0.00,
    net_disposable_income DECIMAL(15,2) GENERATED ALWAYS AS ((monthly_salary + business_income + rental_income + remittance_income + agricultural_income + investment_income + other_income) - total_monthly_expense),
    source_of_funds VARCHAR(150),
    source_of_wealth VARCHAR(150),
    tax_bracket VARCHAR(20),
    declared_net_worth DECIMAL(15,2) DEFAULT 0.00,

    customer_since DATE NOT NULL,
    account_type ENUM('Savings', 'Current', 'FD') DEFAULT 'Savings',
    average_balance DECIMAL(15,2) DEFAULT 0.00,
    monthly_credit_turnover DECIMAL(15,2) DEFAULT 0.00,
    monthly_debit_turnover DECIMAL(15,2) DEFAULT 0.00,
    digital_banking_status ENUM('Enabled', 'Disabled') DEFAULT 'Disabled',
    mobile_banking_status ENUM('Active', 'Inactive') DEFAULT 'Inactive',
    internet_banking_status ENUM('Active', 'Inactive') DEFAULT 'Inactive',
    debit_card_status ENUM('Active', 'Inactive') DEFAULT 'Inactive',

    kyc_status ENUM('Complete', 'Pending', 'Rejected') DEFAULT 'Pending',
    kyc_completion_date DATE,
    risk_rating ENUM('Low', 'Medium', 'High') DEFAULT 'Medium',
    pep_flag ENUM('Yes', 'No') DEFAULT 'No',
    aml_flag ENUM('Yes', 'No') DEFAULT 'No',
    sanctions_match_flag ENUM('Yes', 'No') DEFAULT 'No',
    adverse_media_flag ENUM('Yes', 'No') DEFAULT 'No',
    customer_risk_score INT DEFAULT 0,
    high_risk_customer_flag ENUM('Yes', 'No') GENERATED ALWAYS AS (CASE WHEN risk_rating = 'High' OR pep_flag = 'Yes' OR aml_flag = 'Yes' THEN 'Yes' ELSE 'No' END),
    last_kyc_review_date DATE,
    next_kyc_due_date DATE,

    loan_customer_flag ENUM('Yes', 'No') DEFAULT 'No',
    loan_account_number VARCHAR(30) UNIQUE,
    loan_type ENUM('Home Loan', 'Auto Loan', 'Personal Loan', 'Business Loan', 'SME Loan', 'None') DEFAULT 'None',
    loan_amount DECIMAL(15,2) DEFAULT 0.00,
    approved_amount DECIMAL(15,2) DEFAULT 0.00,
    disbursement_date DATE,
    loan_tenure_months INT DEFAULT 0,
    interest_rate DECIMAL(5,2) DEFAULT 0.00,
    emi_amount DECIMAL(15,2) DEFAULT 0.00,
    outstanding_balance DECIMAL(15,2) DEFAULT 0.00,
    overdue_amount DECIMAL(15,2) DEFAULT 0.00,
    days_past_due INT DEFAULT 0,
    collateral_type VARCHAR(100),
    collateral_value DECIMAL(15,2) DEFAULT 0.00,
    guarantor_name VARCHAR(150),
    guarantor_citizenship VARCHAR(50),

    loan_purpose_code ENUM(
        'HOME_PURCHASE', 'HOME_CONSTRUCTION', 'HOME_RENOVATION', 'LAND_PURCHASE',
        'VEHICLE_PURCHASE', 'EDUCATION', 'MEDICAL', 'PERSONAL_CONSUMPTION',
        'BUSINESS_WORKING_CAPITAL', 'BUSINESS_EXPANSION', 'AGRICULTURE',
        'EQUIPMENT_PURCHASE', 'SME_FINANCING', 'FOREIGN_STUDY', 'MARRIAGE_EXPENSE',
        'DEBT_CONSOLIDATION', 'NONE'
    ) DEFAULT 'NONE',

    gross_monthly_income DECIMAL(15,2) GENERATED ALWAYS AS (monthly_salary + business_income + rental_income),
    net_monthly_income DECIMAL(15,2) GENERATED ALWAYS AS ((monthly_salary + business_income + rental_income + remittance_income + agricultural_income + investment_income + other_income) - total_monthly_expense),
    existing_emi DECIMAL(15,2) DEFAULT 0.00,
    proposed_emi DECIMAL(15,2) DEFAULT 0.00,
    total_emi_obligation DECIMAL(15,2) GENERATED ALWAYS AS (existing_emi + proposed_emi),
    debt_service_ratio DECIMAL(5,2) GENERATED ALWAYS AS (CASE WHEN (monthly_salary + business_income + rental_income) > 0 THEN ((existing_emi + proposed_emi) / (monthly_salary + business_income + rental_income)) * 100 ELSE 0.00 END),
    debt_to_income_ratio DECIMAL(5,2) GENERATED ALWAYS AS (CASE WHEN (monthly_salary + business_income + rental_income) > 0 THEN (existing_emi / (monthly_salary + business_income + rental_income)) * 100 ELSE 0.00 END),
    affordability_score INT DEFAULT 0,
    credit_bureau_score INT DEFAULT 0,
    repayment_capacity DECIMAL(15,2) GENERATED ALWAYS AS (CASE WHEN ((monthly_salary + business_income + rental_income) - (existing_emi + total_monthly_expense)) > 0 THEN ((monthly_salary + business_income + rental_income) - (existing_emi + total_monthly_expense)) ELSE 0.00 END)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""

# View exposing computed `age` so the NL->SQL layer can query it like a column
VIEW = """
CREATE OR REPLACE VIEW customer_kyc_v AS
SELECT k.*, TIMESTAMPDIFF(YEAR, k.date_of_birth, CURDATE()) AS age
FROM customer_kyc k
"""

INSERT_COLUMNS = (
    "customer_id, account_number, customer_type, customer_name, first_name, middle_name, last_name, "
    "nepali_name, gender, date_of_birth, marital_status, nationality, resident_status, citizenship_number, "
    "citizenship_issue_date, citizenship_issue_district, pan_number, mobile_number, email_address, "
    "permanent_province, permanent_district, permanent_municipality, permanent_ward_no, permanent_address, "
    "current_province, current_district, current_municipality, current_ward_no, current_address, "
    "employment_status, occupation, employer_name, monthly_salary, rental_income, business_income, "
    "total_monthly_expense, source_of_funds, customer_since, account_type, average_balance, kyc_status, "
    "risk_rating, loan_customer_flag, loan_type, loan_amount, approved_amount, loan_tenure_months, "
    "interest_rate, emi_amount, loan_purpose_code, existing_emi, proposed_emi, credit_bureau_score"
)

ROWS = [
    ('CIF001234', 'ACC10002001', 'Individual', 'Ram Bahadur Sharma', 'Ram', 'Bahadur', 'Sharma', 'राम बहादुर शर्मा', 'Male', '1988-05-12', 'Married', 'Nepali', 'Resident', '12-01-76-12345', '2009-04-10', 'Kathmandu', '601245789', '9851011111', 'ram.sharma@gmail.com', 'Bagmati', 'Kathmandu', 'Kathmandu Metropolitan', 32, 'Koteshwor', 'Bagmati', 'Kathmandu', 'Kathmandu Metropolitan', 32, 'Koteshwor', 'Salaried', 'Software Engineer', 'Nabil Bank Ltd', 150000.00, 25000.00, 0.00, 65000.00, 'Salary and Rental Income', '2018-01-15', 'Savings', 450000.00, 'Complete', 'Low', 'Yes', 'Home Loan', 10000000.00, 10000000.00, 240, 9.75, 94000.00, 'HOME_PURCHASE', 0.00, 94000.00, 780),
    ('CIF001235', 'ACC10002002', 'Individual', 'Sita Kumari Thapa', 'Sita', 'Kumari', 'Thapa', 'सीता कुमारी थापा', 'Female', '1992-08-24', 'Married', 'Nepali', 'Resident', '27-02-74-09876', '2011-09-15', 'Lalitpur', '602356891', '9841222222', 'sita.thapa@outlook.com', 'Bagmati', 'Lalitpur', 'Lalitpur Metropolitan', 4, 'Jhamsikhel', 'Bagmati', 'Lalitpur', 'Lalitpur Metropolitan', 4, 'Jhamsikhel', 'Salaried', 'Civil Servant', 'Government of Nepal', 65000.00, 0.00, 0.00, 30000.00, 'Salary', '2019-03-22', 'Savings', 120000.00, 'Complete', 'Low', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 710),
    ('CIF001236', 'ACC10002003', 'Individual', 'Hari Prasad Joshi', 'Hari', 'Prasad', 'Joshi', 'हरि प्रसाद जोशी', 'Male', '1975-01-30', 'Married', 'Nepali', 'Resident', '45-01-68-34561', '1994-02-18', 'Kaski', '600112233', '9856033333', 'hari.joshi@yahoo.com', 'Gandaki', 'Kaski', 'Pokhara Metropolitan', 6, 'Lakeside', 'Gandaki', 'Kaski', 'Pokhara Metropolitan', 6, 'Lakeside', 'Self-employed', 'Hotelier', 'Hotel Pokhara Peace', 0.00, 0.00, 450000.00, 120000.00, 'Business Profit', '2015-06-10', 'Current', 1850000.00, 'Complete', 'Medium', 'Yes', 'Business Loan', 25000000.00, 25000000.00, 120, 11.50, 350000.00, 'BUSINESS_EXPANSION', 0.00, 0.00, 740),
    ('CIF001237', 'ACC10002004', 'Individual', 'Nabina Shrestha', 'Nabina', None, 'Shrestha', 'नबिना श्रेष्ठ', 'Female', '1995-11-05', 'Single', 'Nepali', 'Resident', '21-01-72-44321', '2014-12-01', 'Bhaktapur', '605991122', '9803444444', 'nabina.shres@gmail.com', 'Bagmati', 'Bhaktapur', 'Suryabinayak Municipality', 5, 'Katunje', 'Bagmati', 'Bhaktapur', 'Suryabinayak Municipality', 5, 'Katunje', 'Salaried', 'HR Manager', 'Leapfrog Technology', 95000.00, 0.00, 0.00, 40000.00, 'Salary', '2021-07-19', 'Savings', 320000.00, 'Complete', 'Low', 'Yes', 'Auto Loan', 3500000.00, 3200000.00, 60, 10.25, 68000.00, 'VEHICLE_PURCHASE', 0.00, 68000.00, 765),
    ('CIF001238', 'ACC10002005', 'Individual', 'Rajesh Kumar Mahato', 'Rajesh', 'Kumar', 'Mahato', 'राजेश कुमार महतो', 'Male', '1983-03-15', 'Married', 'Nepali', 'Resident', '33-01-70-00124', '2002-05-20', 'Dhanusha', '603445566', '9844055555', 'rajesh.mahato@gmail.com', 'Madhesh', 'Dhanusha', 'Janakpur Metropolitan', 8, 'Mills Area', 'Madhesh', 'Dhanusha', 'Janakpur Metropolitan', 8, 'Mills Area', 'Self-employed', 'Wholesaler', 'Mahato Traders', 0.00, 0.00, 180000.00, 90000.00, 'Agri and Wholesale business', '2017-11-02', 'Current', 890000.00, 'Complete', 'Medium', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 15000.00, 0.00, 690),
    ('CIF001239', 'ACC10002006', 'Individual', 'Anjali Kumari Yadav', 'Anjali', 'Kumari', 'Yadav', 'अञ्जली कुमारी यादव', 'Female', '1998-09-20', 'Single', 'Nepali', 'Resident', '14-02-75-06789', '2016-10-11', 'Morang', '609887766', '9813066666', 'anjali.yadav@outlook.com', 'Koshi', 'Morang', 'Biratnagar Metropolitan', 11, 'Main Road', 'Bagmati', 'Kathmandu', 'Kathmandu Metropolitan', 10, 'Baneshwor', 'Salaried', 'Medical Officer', 'Civil Hospital', 110000.00, 0.00, 0.00, 50000.00, 'Salary', '2023-01-10', 'Savings', 150000.00, 'Complete', 'Low', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 720),
    ('CIF001240', 'ACC10002007', 'Individual', 'Deepak Bahadur Rana', 'Deepak', 'Bahadur', 'Rana', 'दीपक बहादुर राना', 'Male', '1990-07-14', 'Married', 'Nepali', 'Resident', '50-06-73-11223', '2009-08-25', 'Rupandehi', '601223344', '9857077777', 'deepak.rana@gmail.com', 'Lumbini', 'Rupandehi', 'Butwal Sub-Metropolitan', 3, 'Golpark', 'Lumbini', 'Rupandehi', 'Butwal Sub-Metropolitan', 3, 'Golpark', 'Salaried', 'Branch Manager', 'Nepal Telecom', 85000.00, 15000.00, 0.00, 45000.00, 'Salary', '2016-04-05', 'Savings', 540000.00, 'Complete', 'Low', 'Yes', 'Personal Loan', 1000000.00, 1000000.00, 36, 12.00, 33200.00, 'PERSONAL_CONSUMPTION', 0.00, 33200.00, 740),
    ('CIF001241', 'ACC10002008', 'Individual', 'Pooja Karki', 'Pooja', None, 'Karki', 'पूजा कार्की', 'Female', '1994-02-28', 'Married', 'Nepali', 'Resident', '12-02-72-99887', '2012-03-01', 'Kathmandu', '602114455', '9841088888', 'pooja.karki@gmail.com', 'Bagmati', 'Kathmandu', 'Budhanilkantha Municipality', 8, 'Hattigauda', 'Bagmati', 'Kathmandu', 'Budhanilkantha Municipality', 8, 'Hattigauda', 'Salaried', 'Product Designer', 'Cotiviti Nepal', 135000.00, 0.00, 0.00, 60000.00, 'Salary', '2020-09-14', 'Savings', 280000.00, 'Complete', 'Low', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 755),
    ('CIF001242', 'ACC10002009', 'Individual', 'Prem Bahadur Tamang', 'Prem', 'Bahadur', 'Tamang', 'प्रेम बहादुर तामाङ', 'Male', '1981-12-05', 'Married', 'Nepali', 'Resident', '24-01-65-88776', '2000-01-22', 'Kavrepalanchok', '604332211', '9860099999', 'prem.tamang@gmail.com', 'Bagmati', 'Kavrepalanchok', 'Dhulikhel Municipality', 3, 'Bhattedanda', 'Bagmati', 'Kathmandu', 'Bouddha', 6, 'Chabahil', 'Self-employed', 'Contractor', 'Tamang Construction', 0.00, 0.00, 500000.00, 150000.00, 'Business Revenue', '2014-02-28', 'Current', 2100000.00, 'Complete', 'Medium', 'Yes', 'SME Loan', 40000000.00, 35000000.00, 180, 11.00, 420000.00, 'SME_FINANCING', 50000.00, 420000.00, 710),
    ('CIF001243', 'ACC10002010', 'Individual', 'Bina Choudhary', 'Bina', None, 'Choudhary', 'बिना चौधरी', 'Female', '1996-04-17', 'Single', 'Nepali', 'Resident', '64-01-74-12456', '2015-05-19', 'Dang', '607448899', '9808012345', 'bina.chaudhary@gmail.com', 'Lumbini', 'Dang', 'Ghorahi Sub-Metropolitan', 10, 'Narayanpur', 'Lumbini', 'Dang', 'Ghorahi Sub-Metropolitan', 10, 'Narayanpur', 'Salaried', 'Bank Officer', 'Rastriya Banijya Bank', 55000.00, 0.00, 0.00, 22000.00, 'Salary', '2022-11-15', 'Savings', 95000.00, 'Complete', 'Low', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 730),
    ('CIF001244', 'ACC10002011', 'Individual', 'Santosh Kumar Shah', 'Santosh', 'Kumar', 'Shah', 'सन्तोष कुमार शाह', 'Male', '1978-06-25', 'Married', 'Nepali', 'Resident', '16-01-60-33445', '1997-07-12', 'Saptari', '601998877', '9852023456', 'santosh.shah@gmail.com', 'Koshi', 'Saptari', 'Rajbiraj Municipality', 4, 'Main Bazar', 'Koshi', 'Saptari', 'Rajbiraj Municipality', 4, 'Main Bazar', 'Self-employed', 'Pharmacist', 'Shah Medical Hall', 0.00, 40000.00, 250000.00, 80000.00, 'Pharmacy Sales & Rental', '2013-08-09', 'Current', 1450000.00, 'Complete', 'Medium', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 725),
    ('CIF001245', 'ACC10002012', 'Individual', 'Niranjan Raj Bhandari', 'Niranjan', 'Raj', 'Bhandari', 'निरञ्जन राज भण्डारी', 'Male', '1985-10-10', 'Married', 'Nepali', 'Resident', '12-03-68-11224', '2004-11-03', 'Kathmandu', '602447711', '9851045678', 'niranjan.bhandari@outlook.com', 'Bagmati', 'Kathmandu', 'Kathmandu Metropolitan', 2, 'Lazimpat', 'Bagmati', 'Kathmandu', 'Kathmandu Metropolitan', 2, 'Lazimpat', 'Salaried', 'Director', 'UN Agency', 350000.00, 0.00, 0.00, 120000.00, 'Salary', '2012-05-14', 'Savings', 4500000.00, 'Complete', 'High', 'Yes', 'Home Loan', 15000000.00, 15000000.00, 120, 9.50, 194000.00, 'HOME_PURCHASE', 0.00, 0.00, 790),
    ('CIF001246', 'ACC10002013', 'Individual', 'Aayusha Adhikari', 'Aayusha', None, 'Adhikari', 'आयुषा अधिकारी', 'Female', '1999-01-15', 'Single', 'Nepali', 'Resident', '22-01-76-99001', '2017-02-20', 'Chitwan', '608552233', '9865056789', 'aayusha.adh@gmail.com', 'Bagmati', 'Chitwan', 'Bharatpur Metropolitan', 10, 'Hakimchowk', 'Bagmati', 'Kathmandu', 'Kathmandu Metropolitan', 29, 'Anamnagar', 'Salaried', 'Content Strategist', 'F1Soft International', 75000.00, 0.00, 0.00, 35000.00, 'Salary', '2023-08-20', 'Savings', 180000.00, 'Complete', 'Low', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 740),
    ('CIF001247', 'ACC10002014', 'Individual', 'Tek Bahadur Gurung', 'Tek', 'Bahadur', 'Gurung', 'टेक बहादुर गुरुङ', 'Male', '1968-04-05', 'Married', 'Nepali', 'Resident', '46-02-50-12399', '1987-05-14', 'Syangja', '601334499', '9856071234', 'tek.gurung@yahoo.com', 'Gandaki', 'Syangja', 'Waling Municipality', 2, 'Waling Bazar', 'Gandaki', 'Kaski', 'Pokhara Metropolitan', 8, 'Srijana Chowk', 'Retired', 'Ex-Army / Pensioner', 'British Army Pension', 0.00, 20000.00, 0.00, 45000.00, 'Pension and Rental Income', '2010-01-11', 'Savings', 2500000.00, 'Complete', 'Low', 'No', 'None', 0.00, 0.00, 0, 0.00, 0.00, 'NONE', 0.00, 0.00, 760),
    ('CIF001248', 'ACC10002015', 'Individual', 'Sunita Sunuwar', 'Sunita', None, 'Sunuwar', 'सुनीता सुनुवार', 'Female', '1993-12-12', 'Married', 'Nepali', 'Resident', '14-01-71-00991', '2012-01-15', 'Morang', '603881122', '9818098765', 'sunita.sun@gmail.com', 'Koshi', 'Morang', 'Urlabari Municipality', 4, 'Urlabari', 'Bagmati', 'Lalitpur', 'Mahalaxmi Municipality', 2, 'Imadol', 'Salaried', 'Quality Analyst', 'Deerwalk Inc.', 90000.00, 0.00, 0.00, 45000.00, 'Salary', '2021-02-14', 'Savings', 290000.00, 'Complete', 'Low', 'Yes', 'Personal Loan', 500000.00, 500000.00, 24, 12.50, 23700.00, 'MEDICAL', 0.00, 23700.00, 715),
]


def main() -> None:
    conn = mysql.connector.connect(host="localhost", port=3306, user="root", password="", connection_timeout=5)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {TARGET_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cur.execute(f"USE {TARGET_DB}")
    cur.execute(SCHEMA)
    cur.execute(VIEW)

    copied_legacy = 0
    try:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = 'customer_kyc'
            """,
            (LEGACY_DB,),
        )
        legacy_exists = bool(cur.fetchone()[0])
        if legacy_exists:
            cur.execute(
                f"""
                INSERT IGNORE INTO {TARGET_DB}.customer_kyc ({INSERT_COLUMNS})
                SELECT {INSERT_COLUMNS}
                FROM {LEGACY_DB}.customer_kyc
                """
            )
            copied_legacy = cur.rowcount if cur.rowcount is not None else 0
            conn.commit()
    except Exception as exc:
        print(f"legacy migration skipped: {exc}")

    placeholders = ", ".join(["%s"] * 53)
    sql = f"INSERT IGNORE INTO customer_kyc ({INSERT_COLUMNS}) VALUES ({placeholders})"
    cur.executemany(sql, ROWS)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM customer_kyc")
    print(f"customer_kyc rows: {cur.fetchone()[0]}")
    if copied_legacy:
        print(f"copied legacy rows from {LEGACY_DB}: {copied_legacy}")
    cur.execute("SELECT customer_name, age, total_monthly_income, high_risk_customer_flag FROM customer_kyc_v LIMIT 3")
    for row in cur.fetchall():
        print(row)
    conn.close()
    print("kyc_db setup complete.")


if __name__ == "__main__":
    main()
