# Практическая работа №1  
## Установка корпоративной wiki и трекера задач

---

### 2. Установка и настройка DokuWiki

Команда запуска:

```bash
docker run -d --name dokuwiki -p 8080:80 -e PUID=1000 -e PGID=1000 -e TZ=Europe/Moscow -v dokuwiki_data:/config lscr.io/linuxserver/dokuwiki:latest
````

Проверка, что контейнер запущен:

```bash
docker ps
```

![](https://github.com/jamanuriyeva/5sem-4dis/blob/8881b59c00b96f833eeaf285236826bebd103883/TechDocSys/PR1%20wiki/pics/%D1%82%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D0%B0%D0%BB%20%D1%83%D1%81%D1%82%D0%B0%D0%BD%D0%BE%D0%B2%D0%BA%D0%B0.png)


#### 2.2. Доступ к Wiki

После запуска DokuWiki стала доступна по адресу:

```
http://localhost:8080
```

![](https://github.com/jamanuriyeva/5sem-4dis/blob/8881b59c00b96f833eeaf285236826bebd103883/TechDocSys/PR1%20wiki/pics/%D0%BB%D0%BE%D0%BA%D0%B0%D0%BB%D1%85%D0%BE%D1%81%D1%82.png)

---

### 3. Установка и настройка Redmine

#### 3.1 Запуск контейнера Redmine

```bash
docker run -d --name redmine -p 3000:3000 redmine
```

После запуска:

```
http://localhost:3000
```

Авторизация:

```
Логин: admin
Пароль: admin
```

![](https://github.com/jamanuriyeva/5sem-4dis/blob/42495c13971c3acea3a6bc6dfd8bac553c352886/TechDocSys/PR1%20wiki/pics/terminal.png)


![](https://github.com/jamanuriyeva/5sem-4dis/blob/42495c13971c3acea3a6bc6dfd8bac553c352886/TechDocSys/PR1%20wiki/pics/%D0%BF%D1%80%D0%BE%D0%B5%D0%BA%D1%82.png)


### 4. Взаимные ссылки между Wiki и Redmine

#### Пример ссылки из Wiki на задачу Redmine:

```
См. задачу в трекере:
http://localhost:3000/issues/1
```

#### Пример ссылки из задачи Redmine на страницу Wiki:

```
Документация:
http://localhost:8080/doku.php?id=proekt:opisanie
```

Скриншоты:

![Ссылка из Wiki в Redmine](https://github.com/jamanuriyeva/5sem-4dis/blob/42495c13971c3acea3a6bc6dfd8bac553c352886/TechDocSys/PR1%20wiki/pics/%D0%B2%D0%B8%D0%BA%D0%B8.png)
![Ссылка из Redmine в Wiki](https://github.com/jamanuriyeva/5sem-4dis/blob/42495c13971c3acea3a6bc6dfd8bac553c352886/TechDocSys/PR1%20wiki/pics/%D1%80%D0%B5%D0%BC%D0%B8%D0%BD%D0%B8.png)


---

### 5. Анализ возможностей систем

| Система  | Преимущества                                                              | Возможные ограничения                    |
| -------- | ------------------------------------------------------------------------- | ---------------------------------------- |
| DokuWiki | Простая установка, не требует базы данных, удобное редактирование страниц | Ограниченный визуальный редактор         |
| Redmine  | Гибкая система задач, статусы, приоритеты, роли пользователей             | Для расширения функционала нужны плагины |

---

### 6. Заключение

В ходе работы была установлена и настроена корпоративная wiki и система управления задачами. Созданы страницы и задачи, продемонстрировано их взаимное использование через перекрестные ссылки. Такие системы могут эффективно применяться для управления документацией и отслеживания задач внутри команды.

---

### 7. Список источников

1. DokuWiki Installation Guide — [https://www.dokuwiki.org/install](https://www.dokuwiki.org/install)
2. Redmine Installation Guide — [https://www.redmine.org/projects/redmine/wiki/RedmineInstall](https://www.redmine.org/projects/redmine/wiki/RedmineInstall)

```

