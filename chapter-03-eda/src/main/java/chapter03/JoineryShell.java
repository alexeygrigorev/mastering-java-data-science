package chapter03;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import joinery.DataFrame;
import joinery.impl.Shell;

public class JoineryShell {

    public static void main(String[] args) throws IOException {
        List<RankedPage> pages = Data.readRankedPages();
        DataFrame<Object> dataFrame = BeanToJoinery.convert(pages, RankedPage.class);
        Shell.repl(Arrays.asList(dataFrame));
    }
}
