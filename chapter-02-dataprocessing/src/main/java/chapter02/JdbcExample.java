package chapter02;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import javax.sql.DataSource;

import org.apache.commons.collections4.ListUtils;

import com.google.common.collect.Lists;
import com.mysql.jdbc.jdbc2.optional.MysqlDataSource;

public class JdbcExample {

    public static void main(String[] args) throws Exception {
        DataSource datasource = datasource();

        List<Person> list1 = CommonsCsvExample.csvExample();
        List<Person> list2 = CommonsCsvExample.tsvExample();
        List<Person> people = ListUtils.union(list1, list2);
        boolean insert = false;

        if (insert) {
            boolean batch = true;
            if (batch) {
                insertBatch(datasource, people);
            } else {
                insertSimple(datasource, people);
            }
        }

        List<Person> peopleOfGreenland = select(datasource, "Greenland");
        System.out.println(peopleOfGreenland);
    }

    private static List<Person> select(DataSource datasource, String country) throws SQLException {
        try (Connection connection = datasource.getConnection()) {
            String sql = "SELECT name, email, salary, experience FROM people WHERE country = ?;";
            try (PreparedStatement statement = connection.prepareStatement(sql)) {
                List<Person> result = new ArrayList<>();

                statement.setString(1, country);
                try (ResultSet rs = statement.executeQuery()) {
                    while (rs.next()) {
                        String name = rs.getString(1);
                        String email = rs.getString(2);
                        int salary = rs.getInt(3);
                        int experience = rs.getInt(4);
                        Person person = new Person(name, email, country, salary, experience);
                        result.add(person);
                    }
                }

                return result;
            }
        }
    }

    private static void insertSimple(DataSource datasource, List<Person> people) throws SQLException {
        try (Connection connection = datasource.getConnection()) {
            String sql = "INSERT INTO people (name, email, country, salary, experience) VALUES (?, ?, ?, ?, ?);";
            try (PreparedStatement statement = connection.prepareStatement(sql)) {
                for (Person person : people) {
                    statement.setString(1, person.getName());
                    statement.setString(2, person.getEmail());
                    statement.setString(3, person.getCountry());
                    statement.setInt(4, person.getSalary());
                    statement.setInt(5, person.getExperience());
                    statement.execute();
                }
            }
        }
    }

    private static void insertBatch(DataSource datasource, List<Person> people) throws SQLException {
        List<List<Person>> chunks = Lists.partition(people, 50);

        try (Connection connection = datasource.getConnection()) {
            String sql = "INSERT INTO people (name, email, country, salary, experience) VALUES (?, ?, ?, ?, ?);";
            try (PreparedStatement statement = connection.prepareStatement(sql)) {
                for (List<Person> chunk : chunks) {
                    for (Person person : chunk) {
                        statement.setString(1, person.getName());
                        statement.setString(2, person.getEmail());
                        statement.setString(3, person.getCountry());
                        statement.setInt(4, person.getSalary());
                        statement.setInt(5, person.getExperience());
                        statement.addBatch();
                    }
                    statement.executeBatch();
                }
            }
        }
    }

    private static DataSource datasource() {
        MysqlDataSource datasource = new MysqlDataSource();
        datasource.setServerName("localhost");
        datasource.setDatabaseName("people");
        datasource.setUser("root");
        datasource.setPassword("");
        return datasource;
    }
}
