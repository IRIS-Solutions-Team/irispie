
series{
    start=$(series_start)
    data=(
$(series_data)
    )
    period=$(series_period)
    decimals=5
    precision=5
}
 
transform{
    function=$(transform_function)
}
 
x11{
    mode=$(x11_mode)
    save=$(x11_save)
}

