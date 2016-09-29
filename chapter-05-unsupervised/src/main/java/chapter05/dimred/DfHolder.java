package chapter05.dimred;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import java.util.stream.Collectors;

import com.google.common.collect.Lists;

import joinery.DataFrame;

class DfHolder implements Serializable {
    private final List<List<Object>> data;
    private final Collection<Object> index;
    private final Collection<Object> columns;

    public DfHolder(DataFrame<Object> df) {
        this.data = wrap(df.itercols());
        this.index = Lists.newArrayList(df.index());
        this.columns = Lists.newArrayList(df.columns());
    }

    private static List<List<Object>> wrap(ListIterator<List<Object>> cols) {
        return Lists.newArrayList(cols).stream()
                .map(Lists::newArrayList)
                .collect(Collectors.toList());
    }

    public DataFrame<Object> toDf() {
        return new DataFrame<>(index, columns, data);
    }
}