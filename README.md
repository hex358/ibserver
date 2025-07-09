<h1>🗄️InfBlockServer</h1>
🌐 A python server for my Godot Minecraft clone (InfBlock).<br>
<br>Hosted on https://infblock.alwaysdata.net.
<br><br>
List of features:
<ul>
<li>🛢️ player UIDs storage, secured by sha256-encrypted password </li> 
<li>🖼️ skin compression and containment</li>
FastAPI and MongoDB were used.
</ul>
<br>
Even though client save files are stored on hosts device during multiplayer, I didn't want to store player IDs locally - that would have made them easily modifiable.
<br><br>
For this reason I made a separate server. It stores username/password pairs in a MongoDB database, returning players ID when they send a post request.
<br><br>
There is also skin system. <br>
They have to undergo multiple layers of compression though. 
<br>
Palette -> Rectangulation -> Byte packing -> LZMA compression
<br>Thus, we shrink image size from 5KB in Base64 to just 700 bytes.