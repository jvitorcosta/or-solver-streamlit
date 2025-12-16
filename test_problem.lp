maximize 8*cat_food + 10*cat_toys

subject to:
    0.5*cat_food + 0.5*cat_toys <= 150
    cat_food >= 30

where:
    cat_food, cat_toys >= 0