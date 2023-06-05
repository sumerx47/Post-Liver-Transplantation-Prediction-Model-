
-----------------------in table -> query  tool 


-- Copy the data from the CSV file into the table
copy df7 from 'C:\Program Files\PostgreSQL\16\LiverT_dataset.csv' delimiter ',' csv header


select * from df7

select distinct * from df7;



--print Null values
DO $$
DECLARE
  col_name text;
  total_null_values integer := 0;
BEGIN
  -- Loop through each column in the table
  FOR col_name IN (
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'df7'
      AND table_schema = 'public'
  )
  LOOP
    EXECUTE format('
      SELECT COUNT(*) - COUNT(%I) FROM df7;
    ', col_name) INTO total_null_values;
    
    RAISE NOTICE 'Column: %, Total Null Values: %', col_name, total_null_values;
  END LOOP;
END $$;


------------------- drop columns
ALTER TABLE df7
DROP COLUMN column1;



-- Delete rows with null values in specific columns
DELETE FROM df7
WHERE Donor_Hepatitis_B IS NULL OR Recipient_Etiology IS NULL OR Recipient_Gender IS NULL OR Recipient_bmi IS NULL OR Recipient_lympochyte IS NULL OR Recipient_blood_transfusion IS NULL;



-----outlier analysis 
--CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

DROP FUNCTION IF EXISTS handle_outliers(col_name text, tbl_name text);

CREATE OR REPLACE FUNCTION handle_outliers(col_name text, tbl_name text) RETURNS VOID AS $$
DECLARE
  lower_bound numeric;
  upper_bound numeric;
  column_type text;
BEGIN
  -- Get the data type of the column
  EXECUTE format('
    SELECT data_type
    FROM information_schema.columns
    WHERE table_name = %L
      AND column_name = %L
  ', tbl_name, col_name) INTO column_type;

  -- Check if the column type is numeric
  IF column_type = 'numeric' THEN
    -- Calculate the lower and upper bounds using the IQR method
    EXECUTE format('
      SELECT 
        percentile_disc(0.25) WITHIN GROUP (ORDER BY %I) AS q1,
        percentile_disc(0.75) WITHIN GROUP (ORDER BY %I) AS q3
      FROM %I
    ', col_name, col_name, tbl_name) INTO lower_bound, upper_bound;

    -- Apply outlier treatment by replacing values outside the bounds with NULL
    EXECUTE format('
      UPDATE %I
      SET %I = NULL
      WHERE %I < %s OR %I > %s
    ', tbl_name, col_name, col_name, lower_bound, col_name, upper_bound);
  END IF;
END;
$$ LANGUAGE plpgsql;

DROP PROCEDURE IF EXISTS handle_all_outliers(tbl_name text);

CREATE OR REPLACE PROCEDURE handle_all_outliers(tbl_name text) AS $$
DECLARE
  col_name text;
BEGIN
  -- Loop through each column in the table
  FOR col_name IN (
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = tbl_name
      AND table_schema = 'public'
  )
  LOOP
    -- Call the outlier treatment function for each column
    EXECUTE 'SELECT handle_outliers($1, $2)' USING col_name, tbl_name;
  END LOOP;
END;
$$ LANGUAGE plpgsql;

CALL handle_all_outliers('df7');



-- Replace NULL values with the median value
UPDATE df7
SET
  Donor_bmi = rectified_Donor_bmi::numeric,
  Recipient_bmi = rectified_Recipient_bmi::numeric,
  Recipient_primary_biliary_cirrhosis = rectified_Recipient_primary_biliary_cirrhosis::numeric,
  Recipient_na = rectified_Recipient_na::numeric,
  Recipient_mg = rectified_Recipient_mg::numeric,
  Recipient_platelets = rectified_Recipient_platelets::numeric,
  Recipient_cold_ischemia_time = rectified_Recipient_cold_ischemia_time::numeric,
  Recipient_warm_ischemia_time = rectified_Recipient_warm_ischemia_time::numeric
FROM (
  SELECT
    CASE
      WHEN Donor_bmi IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Donor_bmi::numeric) FROM df7 WHERE Donor_bmi IS NOT NULL
      )
      ELSE Donor_bmi::numeric
    END AS rectified_Donor_bmi,
    CASE
      WHEN Recipient_bmi IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_bmi::numeric) FROM df7 WHERE Recipient_bmi IS NOT NULL
      )
      ELSE Recipient_bmi::numeric
    END AS rectified_Recipient_bmi,
    CASE
      WHEN Recipient_primary_biliary_cirrhosis IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_primary_biliary_cirrhosis::numeric) FROM df7 WHERE Recipient_primary_biliary_cirrhosis IS NOT NULL
      )
      ELSE Recipient_primary_biliary_cirrhosis::numeric
    END AS rectified_Recipient_primary_biliary_cirrhosis,
    CASE
      WHEN Recipient_na IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_na::numeric) FROM df7 WHERE Recipient_na IS NOT NULL
      )
      ELSE Recipient_na::numeric
    END AS rectified_Recipient_na,
    CASE
      WHEN Recipient_mg IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_mg::numeric) FROM df7 WHERE Recipient_mg IS NOT NULL
      )
      ELSE Recipient_mg::numeric
    END AS rectified_Recipient_mg,
    CASE
      WHEN Recipient_platelets IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_platelets::numeric) FROM df7 WHERE Recipient_platelets IS NOT NULL
      )
      ELSE Recipient_platelets::numeric
    END AS rectified_Recipient_platelets,
    CASE
      WHEN Recipient_cold_ischemia_time IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_cold_ischemia_time::numeric) FROM df7 WHERE Recipient_cold_ischemia_time IS NOT NULL
      )
      ELSE Recipient_cold_ischemia_time::numeric
    END AS rectified_Recipient_cold_ischemia_time,
    CASE
      WHEN Recipient_warm_ischemia_time IS NULL THEN (
        SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY Recipient_warm_ischemia_time::numeric) FROM df7 WHERE Recipient_warm_ischemia_time IS NOT NULL
      )
      ELSE Recipient_warm_ischemia_time::numeric
    END AS rectified_Recipient_warm_ischemia_time
  FROM df7
) AS rectified
WHERE
  df7.Donor_bmi IS NULL
  OR df7.Recipient_bmi IS NULL
  OR df7.Recipient_primary_biliary_cirrhosis IS NULL
  OR df7.Recipient_na IS NULL
  OR df7.Recipient_mg IS NULL
  OR df7.Recipient_platelets IS NULL
  OR df7.Recipient_cold_ischemia_time IS NULL
  OR df7.Recipient_warm_ischemia_time IS NULL;






