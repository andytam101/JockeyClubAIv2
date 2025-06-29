for i in {5..6}
do
    python true_listwise.py location_ST_1000 "$i"
    python true_listwise.py location_ST_1200 "$i"
    python true_listwise.py location_ST_1400 "$i"
    python true_listwise.py location_ST_1600 "$i"
    python true_listwise.py location_ST_1800 "$i"
    python true_listwise.py location_ST_2000 "$i"

    python true_listwise.py location_HV_1000 "$i"
    python true_listwise.py location_HV_1200 "$i"
    python true_listwise.py location_HV_1800 "$i"
    python true_listwise.py location_HV_2200 "$i"
done
