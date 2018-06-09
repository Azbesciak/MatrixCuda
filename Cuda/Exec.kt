import java.io.BufferedReader
import java.io.InputStreamReader

val blockSizes = listOf(8,16,32)
val matSizes = listOf(128, 256,512,1024,2048)
const val matNum = 10

data class Param(val n: Int, val b: Int, val s: Int, val m: Int)

fun main(args: Array<String>) {
    require(args.size == 1) {
        "path to cuda exe requried"
    }
    val path = args[0]
    matSizes.flatMap { n -> blockSizes.flatMap { b -> listOf(Param(n, b, matNum, matNum), Param(n, b, 1, matNum)) } }
            .forEach { execCmd("$path -n=${it.n} -m=${it.m} -s=${it.s} -b=${it.b}") }
}

fun execCmd(cmd: String) {
    val proc = ProcessBuilder(cmd.split(" "))
            .redirectErrorStream(true)
            .start()
    val inputStream = proc.inputStream
    return BufferedReader(InputStreamReader(inputStream)).use {
        val result =  it.lines()
                .peek { println("cmd: $it") }
                .map {
                    when {
                        it.contains("Performance=") -> true
                        else -> null
                    }
                }.filter { it != null }
                .findFirst()
                .orElse(false)
        result!!
    }
}
