# Finance_Graph_Theory

BS_price = function(St, K, r, T, t, sigma)
           {
           d1 = (log(St/K) + (r + (sigma^2)/2) * (T-t))/(sigma * (T-t) ^ 0.5)
           d2 = d1 - sigma * (T-t)^0.5
           call = pnorm(d1) * St - pnorm(d2) * K * 2.73^(r*(t-T))
           print (call)
           }
           BS_price(11377.75, 10000, 0.06, 38/365, 0, 0.05)