import qrcode

url = "https://newsrecommendationsystem-dwx.streamlit.app/"
qr = qrcode.make(url)
qr.save("streamlit_qr.png")
