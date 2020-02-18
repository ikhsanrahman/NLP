import sentiment_mod as s 

data_real = {
	"0": "Hy sara, kamu orang nya baik. tetap jadilah pribadi yang baik! ",
	"1": "kita tidak bisa melawan sekarang. tapi kita harus latihan jadi lebih kuat.",
	"2": "dermawan dan tidak sombong menjadikan manusia lebih disayangi dan dimudahkan rezekinya",
	"3": "Sebelum itu, ada baiknya kita mencari tahu terlebih dahulu apa itu pikiran positif, karena tidak jarang banyak orang yang sudah memahami namun sulit untuk dilakukan",
	"4": "Emang sih ada masanya dimana kita harus bersikap waspada, karena bisa saja segala sesuatu bisa menjerumuskan kita dengan hal-hal yang tidak baik, apalagi jika saat ini kita sedang menghadapi masalah dan cobaan hidup",
	"5": "bersikap waspada juga tidak baik jika dilakukan dengan cara yang berlebihan",
	"6": "Memeilihara pemikiran yang negatif tidak baik untuk perkembangan kita sendiri, karena pemikiran negatif akan menghambat pada pencapaian dan keberhasilan untuk meraih kesuksesan masa depan",
	"7": "Cobalah untuk tidak melihat pada sisi buruk sesuatu dalam waktu yang lama, lebih baik kita mencari solusi yang tepat untuk mengatasi pikiran-pikiran negatif dengan cara yang positif",
	"8": "Tak harus meniru sesuatu untuk menjadi keren. Cukup jadi diri sendiri dan kenali siapa kamu sebenarnya.",
	"9": "Apapun yang terjadi, nikmati hidup ini. Hapus air mata lalu berikan senyumanmu. Kadang, senyum terindah datang setelah air mata penuh luka.",
	"10": '''Bagaimana kita menjaga api batin kita hidup? Dua hal, minimal, diperlukan: kemampuan untuk menghargai positif dalam hidup kita dan komitmen untuk bertindak. Setiap hari, penting untuk bertanya dan menjawab pertanyaan-pertanyaan ini: "Apa yang baik dalam hidup saya?" Dan "Apa yang perlu dilakukan?''',
	"11": "kamu brengsek. tidak tau diri.",
	"12": "laki-laki jelek hanya untuk perempuan yang jelek",
	"13": "sombong sekali kamu sehingga kamu semena mena sama saya",
	"14": "hoi, tolong ambil sampah itu dasar laki-laki gak tau diuntung",
	"15": "jangan pernah balik lagi dasar perempuan jalang",
	"16": "sebaiknya kamu masuk atau ku pukul kamu",
	"17": "makanya mikir pakai otak, jangan sembarang nendang tupperware mak ya",
	"18": "lihat dengan mata jangan lihat makai telinga. wew",
	"19": "Tapi pertanyaanya, bagaimana perasaan kamu jika sedang berpikiran negatif ? Jawabannya mungkin kamu merasakan khawatir yang berlebihan, cemas, gelisah, galau, takut, merasa tidak tenang dan gugup",
}


for i in range(len(data_real)):
	result = s.sentiment(data_real[str(i)])
	print("sentiment dari kalimat yang ke {} adalah {} dengan akurasi {}%".format(i, result[0], result[1]*100))