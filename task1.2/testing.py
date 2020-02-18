import sentiment_mod as s 

data_real = {
	"0": "bajingan kamu, kamu kira bisa mengalahkan aku",
	"1": "hey kamu tidak boleh seperti itu, dengar tu",
	"2": "Ahok mampu melawan korupsi di ibukota, bangun infrastruktur, mencoba memperbaiki dengan berbagai trobosan dan works well",
	"3": "Gila gara2 pilkada, beberapa tokoh menampakkan watak aslinya",
	"4": "Ibu2 aja, kalo beli telor pake milih2 sblm ditimbang Ini Pesan Ahok di Hari Pers Nasional",
	"5": "Serius tapi ada selipan hiburannya dan Sylvie sukses blunder bikin paslon 1 drop elektabilitasnya",
	"6": "tu yang ngomong jangan pilih Ahok Djarot pernah ngerasain banjir ga sih? dan tolong ya filter berita share berita baik dripada hoax",
	"7": "Genderang perang makin kencang dan OK OCE anies dan sandi dalam mengatasi urbanisasi dan lapangan kerja semakin meningkat",
	"8": "Yg no.1 nih ngomongin hal yg di luar program mulu ya. dan Ini judulnya Debat atau nyerang paslon 2 ??????",
	"9": "Ahok keren!!!. hellooo, itu hanya bullshit karena penuh dengan omongan yang busuk",
	"10": "Bravo closing statement Anies malam ini.kalau nggak ada debat saya nggak mungkin nonton",
	"11": "Yg ga punya program bisanya cuma fitnah. Ciri Orang yg demen ngambil kjp kontan dan Mpok silvi nyinyir melulu pertanyaan nya tapi sayang sesat!",
	"12": "Tangkap Ahok @basuki_btp skrng juga dan menurut saya keliatan banget bego nya mpok silvy and agus",
	"13": "Teman Ahok emank TAI. apalagi pemimpinnya kek BABI!",
	"14": "Kampungan ya pendukung AHY",
	"15": "pak anies banyak omdo dulu waktu jd menteri terobosannya apa?? sok kali ini kalau ngomong",
	"16": "Babi... iya memang si Ahok gak jauh kayak Babi. mulutnya penuh dengan kotoran dan kasar sekali",
	"17": "Sepertinya gerombolan @basuki_btp rindu kerusuhan, dan keributan yang tanpa henti hati2 dikabul Tuhan",
	"18": "otak wc otak babi yg menerima ahok kafir. wew aja deh",
	"19": "Kayak gini nih kampanye Tim AniesSandiUno. ke kurangan Ide kali ya. bodoh amat sih",
}


for i in range(len(data_real)):
	result = s.sentiment(data_real[str(i)])
	print("sentiment dari kalimat yang ke {} adalah {} dengan akurasi {}%".format(i, result[0], result[1]*100))