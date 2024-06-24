# script to delete the rows in donwstream_results.csv that have vit in their model_name

file="downstream_results.csv"
# delete the rows that have vit in their model_name
# first count the number of rows that have vit in their model_name
count=$(grep -c "vit" $file)
echo "Number of rows that have vit in their model_name: $count"

# delete the rows that have vit in their model_name
# uncomment after confirming count
# sed -i '/vit/d' $file
echo "Deleted the rows that have vit in their model_name"