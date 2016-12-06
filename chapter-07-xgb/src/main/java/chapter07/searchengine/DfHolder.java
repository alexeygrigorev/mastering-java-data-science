package chapter07.searchengine;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import java.util.stream.Collectors;

import com.google.common.collect.Lists;

import joinery.DataFrame;

class DfHolder<E> implements Serializable {
    private final List<List<E>> data;
    private final Collection<Object> index;
    private final Collection<Object> columns;

    public DfHolder(DataFrame<E> df) {
        this.data = wrap(df.itercols());
        this.index = Lists.newArrayList(df.index());
        this.columns = Lists.newArrayList(df.columns());
    }

    private List<List<E>> wrap(ListIterator<List<E>> cols) {
        return Lists.newArrayList(cols).stream()
                .map(Lists::newArrayList)
                .collect(Collectors.toList());
    }

    public DataFrame<E> toDf() {
        return new DataFrame<>(index, columns, data);
    }
}