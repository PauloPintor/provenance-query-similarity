
select  i_item_id
       ,i_item_desc
       ,i_current_price
       ,array_agg(list_value(item.prov, inventory.prov, date_dim.prov, catalog_sales.prov)) prov
 from item, inventory, date_dim, catalog_sales
 where i_current_price between 70 and 70 + 30
 and inv_item_sk = i_item_sk
 and d_date_sk=inv_date_sk
 and d_date between cast('1999-05-09' as date) and cast('1999-05-09' as date) + INTERVAL 60 DAY
 and i_manufact_id in (884,774,914,927)
 and inv_quantity_on_hand between 100 and 500
 and cs_item_sk = i_item_sk
 group by i_item_id,i_item_desc,i_current_price
 order by i_item_id
 LIMIT 100;


