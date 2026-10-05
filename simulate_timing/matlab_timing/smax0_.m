function a = smax0_(b,c)

    a = b;
    inx = a<0;
    a(inx) = c * a(inx);
end