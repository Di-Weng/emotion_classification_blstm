drop table if exists user_sentiment;
create table user_sentiment(
	id int identity(1,1),
	userName text not null,
	use_date date not null,
	surprise int not null,
	angry int not null,
	sad int not null,
	fear int not null,
	happy int not null
);