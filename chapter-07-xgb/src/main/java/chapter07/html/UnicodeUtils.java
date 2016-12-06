package chapter07.html;

import org.apache.commons.lang3.text.translate.AggregateTranslator;
import org.apache.commons.lang3.text.translate.CharSequenceTranslator;
import org.apache.commons.lang3.text.translate.EntityArrays;
import org.apache.commons.lang3.text.translate.LookupTranslator;

public class UnicodeUtils {

    private static final CharSequence[][] UNICODE_TRANSLATION = { 
            { "\u00A0", " " }, { "\u00AB", "\"" }, { "\u00AD", "-" }, { "\u00B4", "'" }, { "\u00BB", "\"" },
            { "\u00F7", "/" }, { "\u01C0", "|" }, { "\u01C3", "!" }, { "\u02B9", "'" }, { "\u02BA", "\"" },
            { "\u02BC", "'" }, { "\u02C4", "^" }, { "\u02C6", "^" }, { "\u02C8", "'" }, { "\u02CB", "`" },
            { "\u02CD", "_" }, { "\u02DC", "~" }, { "\u0300", "`" }, { "\u0301", "'" }, { "\u0302", "^" },
            { "\u0303", "~" }, { "\u030B", "\"" }, { "\u030E", "\"" }, { "\u0331", "_" }, { "\u0332", "_" },
            { "\u0338", "/" }, { "\u0589", ":" }, { "\u05C0", "|" }, { "\u05C3", ":" }, { "\u066A", "%" },
            { "\u066D", "*" }, { "\u200B", " " }, { "\u2010", "-" }, { "\u2011", "-" }, { "\u2012", "-" },
            { "\u2013", "-" }, { "\u2014", "-" }, { "\u2015", "-" }, { "\u2016", "|" }, { "\u2017", "_" },
            { "\u2018", "'" }, { "\u2019", "'" }, { "\u201A", "," }, { "\u201B", "'" }, { "\u201C", "\"" },
            { "\u201D", "\"" }, { "\u201E", "\"" }, { "\u201F", "\"" }, { "\u2026", "..." }, { "\u2032", "'" },
            { "\u2033", "\"" }, { "\u2034", "'" }, { "\u2035", "`" }, { "\u2036", "\"" }, { "\u2037", "'" },
            { "\u2038", "^" }, { "\u2039", "<" }, { "\u203A", ">" }, { "\u203D", "?" }, { "\u2044", "/" },
            { "\u204E", "*" }, { "\u2052", "%" }, { "\u2053", "~" }, { "\u2060", " " }, { "\u20E5", "\\" },
            { "\u2212", "-" }, { "\u2215", "/" }, { "\u2216", "\\" }, { "\u2217", "*" }, { "\u2223", "|" },
            { "\u2236", ":" }, { "\u223C", "~" }, { "\u2264", "<" }, { "\u2265", ">" }, { "\u2266", "<" },
            { "\u2267", ">" }, { "\u2303", "^" }, { "\u2329", "<" }, { "\u232A", ">" }, { "\u266F", "#" },
            { "\u2731", "*" }, { "\u2758", "|" }, { "\u2762", "!" }, { "\u27E6", "[" }, { "\u27E8", "<" },
            { "\u27E9", ">" }, { "\u2983", "{" }, { "\u2984", "}" }, { "\u3003", "\"" }, { "\u3008", "<" },
            { "\u3009", ">" }, { "\u301B", "]" }, { "\u301C", "~" }, { "\u301D", "\"" }, { "\u301E", "\"" },
            { "\uE100", "!" },  { "\u2048", "?!" }, { "\u202F", " " },
            { "\uFEFF", " " }, { "\uFFFD", " " }, { "\u0004", " " }, { "\u0008", " " }, { "\u0009", " " }, 
            { "\u0009", " " }, { "\u000B", " " }, { "\u000C", " " },
            { String.valueOf((char) 0x000A), " " }, { String.valueOf((char) 0x000D), " " },
            { "\u000E", " " }, { "\u000F", " " }, 
            { "\u0012", " " }, { "\u0015", " " }, { "\u0017", " " }, { "\u0019", " " }, { "\u009B", " " }, 
            { "\u038D", " " }, 
    };

    private static final CharSequenceTranslator TRANSLATOR = new AggregateTranslator(
            new LookupTranslator(UNICODE_TRANSLATION), new LookupTranslator(EntityArrays.ISO8859_1_UNESCAPE()),
            new LookupTranslator(EntityArrays.BASIC_UNESCAPE()),
            new LookupTranslator(EntityArrays.HTML40_EXTENDED_UNESCAPE()));

    public static final String fixUnicode(String input) {
        return TRANSLATOR.translate(input);
    }

}
