package chapter04;

import java.beans.BeanInfo;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import joinery.DataFrame;

public class BeanToJoinery {

    public static <E> DataFrame<Object> convert(List<E> beans, Class<E> beanClass) {
        try {
            return doConvert(beans, beanClass);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static <E> DataFrame<Object> doConvert(List<E> beans, Class<E> beanClass) throws Exception {
        int nrow = beans.size();
        BeanInfo info = Introspector.getBeanInfo(beanClass);
        PropertyDescriptor[] properties = info.getPropertyDescriptors();

        Map<String, List<Object>> columns = Maps.newLinkedHashMap();

        for (PropertyDescriptor pd : properties) {
            String name = pd.getName();
            if ("class".equals(name)) {
                continue;
            }

            Method getter = pd.getReadMethod();
            if (getter == null) {
                continue;
            }

            List<Object> column = Lists.newArrayListWithCapacity(nrow);
            for (E e : beans) {
                Object value = getter.invoke(e);
                column.add(value);
            }

            columns.put(name, column);
        }

        List<Integer> index = IntStream.range(0, nrow)
                .mapToObj(Integer::valueOf)
                .collect(Collectors.toList());

        List<String> columnNames = Lists.newArrayList(columns.keySet());
        List<List<Object>> data = Lists.newArrayList(columns.values());
        return new DataFrame<>(index, columnNames, data);
    }

}
