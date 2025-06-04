<h1>🗄️InfBlockServer</h1>
🌐 A python FastAPI server for my Godot Minecraft clone (InfBlock).<br>
<br>Hosted on <a hred="https://infblock.alwaysdata.net">alwaysdata</a>.
<br><br>
Even though client save files are stored on hosts device during multiplayer, I didn't want to store player IDs locally - that would have made them easily modifiable.
<br><br>
To fix this I made a separate server. It stores username/password pairs in a MongoDB database, returning players ID when they send a post request.
<br><br>
There is also skin system. <br>
They have to undergo multiple layers of compression though. 
<br>
Palette -> Rectangulation -> Byte packing -> LZMA compression