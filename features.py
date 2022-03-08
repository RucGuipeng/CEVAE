STR_FEATURES_V1 = 'gender,age_10,age_5,edu_degree,is_college_student,is_married,is_pregnant,is_has_child,is_marriage_seeking,occupation,work_time_type,business_trip_freq,is_job_seeking,user_group_label,user_group_label_name,mobile_brand,mobile_model,mobile_price,residence_country_name,residence_province_name,residence_city_name,residence_district_name,residence_city_level,is_rent,is_house_purchasing,is_house_decorating,is_own_car,is_car_purchasing,rank1_cnt_30d_first_cate_name,rank2_cnt_30d_first_cate_name,rank3_cnt_30d_first_cate_name,rank1_amt_30d_first_cate_name,rank2_amt_30d_first_cate_name,rank3_amt_30d_first_cate_name,user_active_hour,ad_brand_prefer,ecom_first_cate_prefer,short_video_cate1_prefer,short_video_cate3_prefer,info_cate_prefer,info_keyword_prefer,game_prefer,interest_prefer,activeness_level,consuming_level,platform_consuming_level,ecom_user_level,life_stage,is_aweme_dau,is_aweme_mau,is_douyin_lite_dau,is_douyin_lite_mau,is_live_stream_dau,is_live_stream_mau'.split(',')
STR_FEATURES_V2 = 'potential_conversion_level,second_hand_conversion_level,lifestyle_conversion_level,jewelry_conversion_level,intelligence_conversion_level,gardening_conversion_level,fresh_conversion_level,foods_conversion_level,fashion_conversion_level,drinks_conversion_level,cleansing_conversion_level,beauty_conversion_level,baby_conversion_level,3c_conversion_level,cross_border_conversion_level,ka_conversion_level,first_name_female_cloth_conversion_level,first_name_female_shoes_conversion_level,first_name_mobile_conversion_level,first_name_male_cloth_conversion_level,first_name_male_shoes_conversion_level,first_name_wine_conversion_level,first_name_books_conversion_level,first_name_snack_conversion_level,first_name_antique_conversion_level,first_name_jewelry_conversion_level,first_name_beauty_conversion_level,first_name_baby_conversion_level,first_name_salon_conversion_level,first_name_home_conversion_level,first_name_self_conversion_level,first_name_underwear_conversion_level,first_name_fashion_conversion_level,first_name_care_conversion_level,first_name_foods_conversion_level,first_name_gardening_conversion_level,first_name_health_conversion_level,first_name_paper_conversion_level,first_name_nutritious_conversion_level,sports_conversion_level,beauty_om_conversion_level,beauty_rh_conversion_level,beauty_gh_conversion_level,douyin_lite_potential_conversion_level'.split(',')
STR_FEATURES_V3 = 'avg_cost_cnt_1d_level,avg_cost_cnt_7d_level,avg_cost_cnt_15d_level,avg_cost_cnt_30d_level,avg_send_cnt_1d_level,avg_send_cnt_7d_level,avg_send_cnt_15d_level,avg_send_cnt_30d_level,avg_show_cnt_1d_level,avg_show_cnt_7d_level,avg_show_cnt_15d_level,avg_show_cnt_30d_level,avg_click_cnt_1d_level,avg_click_cnt_7d_level,avg_click_cnt_15d_level,avg_click_cnt_30d_level,avg_convert_cnt_1d_level,avg_convert_cnt_7d_level,avg_convert_cnt_15d_level,avg_convert_cnt_30d_level'.split(',')
STR_FEATURES_V4 = 'home_level,rural_label'.split(',')
FLOAT_FEATURES_V1 = 'live_cart_click_cnt,live_cart_click_cnt_7d,live_cart_click_cnt_30d,watch_cnt,watch_cnt_7d,watch_cnt_30d,video_cart_click_cnt,video_cart_click_cnt_7d,video_cart_click_cnt_30d,wish_button_click_cnt,wish_button_click_cnt_7d,wish_button_click_cnt_30d,share_product_cnt,share_product_cnt_7d,share_product_cnt_30d,pay_cnt,pay_amt,pay_days,pay_cnt_7d,pay_amt_7d,pay_days_7d,pay_cnt_30d,pay_amt_30d,pay_days_30d,total_pay_cnt,total_pay_amt,total_pay_days'.split(',')
FLOAT_FEATURES_V2 = 'new_user_coupon_hist_day_sum30,new_user_coupon_hist_day_mean30,new_user_coupon_hist_day_days30,new_user_coupon_hist_day_rate30,new_user_coupon_hist_day_std30,new_user_coupon_hist_day_max30,new_user_coupon_hist_day_days_mean30,new_user_coupon_hist_day_sum15,new_user_coupon_hist_day_mean15,new_user_coupon_hist_day_days15,new_user_coupon_hist_day_rate15,new_user_coupon_hist_day_std15,new_user_coupon_hist_day_max15,new_user_coupon_hist_day_days_mean15,new_user_coupon_hist_day_sum7,new_user_coupon_hist_day_mean7,new_user_coupon_hist_day_days7,new_user_coupon_hist_day_rate7,new_user_coupon_hist_day_std7,new_user_coupon_hist_day_max7,new_user_coupon_hist_day_days_mean7,new_user_coupon_hist_sum30,new_user_coupon_hist_mean30,new_user_coupon_hist_days30,new_user_coupon_hist_rate30,new_user_coupon_hist_std30,new_user_coupon_hist_max30,new_user_coupon_hist_days_mean30,new_user_coupon_hist_sum15,new_user_coupon_hist_mean15,new_user_coupon_hist_days15,new_user_coupon_hist_rate15,new_user_coupon_hist_std15,new_user_coupon_hist_max15,new_user_coupon_hist_days_mean15,new_user_coupon_hist_sum7,new_user_coupon_hist_mean7,new_user_coupon_hist_days7,new_user_coupon_hist_rate7,new_user_coupon_hist_std7,new_user_coupon_hist_max7,new_user_coupon_hist_days_mean7'.split(',')
FLOAT_FEATURES_V3 = 'potential_model_score,second_hand_model_score,lifestyle_model_score,jewelry_model_score,intelligence_model_score,gardening_model_score,fresh_model_score,foods_model_score,fashion_model_score,drinks_model_score,cleansing_model_score,beauty_model_score,baby_model_score,3c_model_score,cross_border_model_score,ka_model_score,first_name_female_cloth_model_score,first_name_female_shoes_model_score,first_name_mobile_model_score,first_name_male_cloth_model_score,first_name_male_shoes_model_score,first_name_wine_model_score,first_name_books_model_score,first_name_snack_model_score,first_name_antique_model_score,first_name_jewelry_model_score,first_name_beauty_model_score,first_name_baby_model_score,first_name_salon_model_score,first_name_home_model_score,first_name_self_model_score,first_name_underwear_model_score,first_name_fashion_model_score,first_name_care_model_score,first_name_foods_model_score,first_name_gardening_model_score,first_name_health_model_score,first_name_paper_model_score,first_name_nutritious_model_score,sports_model_score,beauty_om_model_score,beauty_rh_model_score,beauty_gh_model_score,douyin_lite_potential_model_score'.split(',')
FLOAT_FEATURES_V4 = 'avg_cost_cnt_1d,avg_cost_cnt_1d_cut,avg_cost_cnt_7d,avg_cost_cnt_7d_cut,avg_cost_cnt_15d,avg_cost_cnt_15d_cut,avg_cost_cnt_30d,avg_cost_cnt_30d_cnt,avg_send_cnt_1d,avg_send_cnt_1d_cut,avg_send_cnt_7d,avg_send_cnt_7d_cut,avg_send_cnt_15d,avg_send_cnt_15d_cut,avg_send_cnt_30d,avg_send_cnt_30d_cut,avg_show_cnt_1d,avg_show_cnt_1d_cut,avg_show_cnt_7d,avg_show_cnt_7d_cut,avg_show_cnt_15d,avg_show_cnt_15d_cut,avg_show_cnt_30d,avg_show_cnt_30d_cut,avg_click_cnt_1d,avg_click_cnt_1d_cut,avg_click_cnt_7d,avg_click_cnt_7d_cut,avg_click_cnt_15d,avg_click_cnt_15d_cut,avg_click_cnt_30d,avg_click_cnt_30d_cut,avg_convert_cnt_1d,avg_convert_cnt_1d_cut,avg_convert_cnt_7d,avg_convert_cnt_7d_cut,avg_convert_cnt_15d,avg_convert_cnt_15d_cut,avg_convert_cnt_30d,avg_convert_cnt_30d_cut'.split(',')
FLOAT_FEATURES_V5 = 'home_price'.split(',')
FLOAT_FEATURES_V6 = 'ecom_show_cnt_rec1d,ecom_watch_cnt_rec1d,ecom_entrance_click_cnt_rec1d,cart_click_cnt_rec1d,product_click_cnt_rec1d,order_submit_show_cnt_rec1d,order_submit_click_cnt_rec1d,order_submit_success_cnt_rec1d,ecom_show_cnt_rec7d,ecom_watch_cnt_rec7d,ecom_entrance_click_cnt_rec7d,cart_click_cnt_rec7d,product_click_cnt_rec7d,order_submit_show_cnt_rec7d,order_submit_click_cnt_rec7d,order_submit_success_cnt_rec7d,ecom_show_cnt_rec14d,ecom_watch_cnt_rec14d,ecom_entrance_click_cnt_rec14d,cart_click_cnt_rec14d,product_click_cnt_rec14d,order_submit_show_cnt_rec14d,order_submit_click_cnt_rec14d,order_submit_success_cnt_rec14d,ecom_show_cnt_rec30d,ecom_watch_cnt_rec30d,ecom_entrance_click_cnt_rec30d,cart_click_cnt_rec30d,product_click_cnt_rec30d,order_submit_show_cnt_rec30d,order_submit_click_cnt_rec30d,order_submit_success_cnt_rec30d,live_ecom_show_cnt_rec1d,live_ecom_watch_cnt_rec1d,live_ecom_entrance_click_cnt_rec1d,live_cart_click_cnt_rec1d,live_product_click_cnt_rec1d,live_order_submit_show_cnt_rec1d,live_order_submit_click_cnt_rec1d,live_order_submit_success_cnt_rec1d,live_ecom_show_cnt_rec7d,live_ecom_watch_cnt_rec7d,live_ecom_entrance_click_cnt_rec7d,live_cart_click_cnt_rec7d,live_product_click_cnt_rec7d,live_order_submit_show_cnt_rec7d,live_order_submit_click_cnt_rec7d,live_order_submit_success_cnt_rec7d,live_ecom_show_cnt_rec14d,live_ecom_watch_cnt_rec14d,live_ecom_entrance_click_cnt_rec14d,live_cart_click_cnt_rec14d,live_product_click_cnt_rec14d,live_order_submit_show_cnt_rec14d,live_order_submit_click_cnt_rec14d,live_order_submit_success_cnt_rec14d,live_ecom_show_cnt_rec30d,live_ecom_watch_cnt_rec30d,live_ecom_entrance_click_cnt_rec30d,live_cart_click_cnt_rec30d,live_product_click_cnt_rec30d,live_order_submit_show_cnt_rec30d,live_order_submit_click_cnt_rec30d,live_order_submit_success_cnt_rec30d,video_ecom_show_cnt_rec1d,video_ecom_watch_cnt_rec1d,video_ecom_entrance_click_cnt_rec1d,video_cart_click_cnt_rec1d,video_product_click_cnt_rec1d,video_order_submit_show_cnt_rec1d,video_order_submit_click_cnt_rec1d,video_order_submit_success_cnt_rec1d,video_ecom_show_cnt_rec7d,video_ecom_watch_cnt_rec7d,video_ecom_entrance_click_cnt_rec7d,video_cart_click_cnt_rec7d,video_product_click_cnt_rec7d,video_order_submit_show_cnt_rec7d,video_order_submit_click_cnt_rec7d,video_order_submit_success_cnt_rec7d,video_ecom_show_cnt_rec14d,video_ecom_watch_cnt_rec14d,video_ecom_entrance_click_cnt_rec14d,video_cart_click_cnt_rec14d,video_product_click_cnt_rec14d,video_order_submit_show_cnt_rec14d,video_order_submit_click_cnt_rec14d,video_order_submit_success_cnt_rec14d,video_ecom_show_cnt_rec30d,video_ecom_watch_cnt_rec30d,video_ecom_entrance_click_cnt_rec30d,video_cart_click_cnt_rec30d,video_product_click_cnt_rec30d,video_order_submit_show_cnt_rec30d,video_order_submit_click_cnt_rec30d,video_order_submit_success_cnt_rec30d,live_confirm_add_to_cart_cnt_rec1d,live_confirm_add_to_cart_cnt_rec7d,live_confirm_add_to_cart_cnt_rec14d,live_confirm_add_to_cart_cnt_rec30d,view_product_avg_price_rec1d,view_product_avg_price_rec7d,view_product_avg_price_rec14d,view_product_avg_price_rec30d'.split(',')
STRING_FEATURES = STR_FEATURES_V1 + STR_FEATURES_V2 + STR_FEATURES_V3 + STR_FEATURES_V4
FLOAT_FEATURES = FLOAT_FEATURES_V1 + FLOAT_FEATURES_V2 + FLOAT_FEATURES_V3 + FLOAT_FEATURES_V4 + FLOAT_FEATURES_V5 + FLOAT_FEATURES_V6
