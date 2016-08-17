
package chapter03;

import java.io.IOException;
import java.util.List;

import joinery.DataFrame;

public class JoineryExample {

    public static void main(String[] args) throws IOException {
        List<RankedPage> pages = Data.readRankedPages();
        DataFrame<Object> df = BeanToJoinery.convert(pages, RankedPage.class);

        DataFrame<Object> drop = df.retain("bodyContentLength", "titleLength", "numberOfHeaders");
        DataFrame<Object> describe = drop.describe();
        System.out.println(describe.toString());

DataFrame<Object> meanPerPage = df.groupBy("page").mean()
        .drop("position")
        .sortBy("page").transpose();
System.out.println(meanPerPage);
    }
}
